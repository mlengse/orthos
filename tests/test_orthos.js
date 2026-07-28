/**
 * Minimal JS test harness for orthos.js
 * Uses child_process.spawn to run the script with test inputs.
 */

const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const assert = require("assert");

const ORTHOS = path.join(__dirname, "..", "orthos.js");
const TMP = path.join(__dirname, "..", ".test_tmp");

function setup() {
    if (!fs.existsSync(TMP)) fs.mkdirSync(TMP, { recursive: true });
}

function teardown() {
    if (fs.existsSync(TMP)) {
        for (const f of fs.readdirSync(TMP)) {
            fs.unlinkSync(path.join(TMP, f));
        }
        fs.rmdirSync(TMP);
    }
}

function writeTmp(name, content) {
    const p = path.join(TMP, name);
    fs.writeFileSync(p, content, "utf-8");
    return p;
}

function runOrthos(dictPath, patInPath, patOutPath, inputs, timeout) {
    return new Promise((resolve, reject) => {
        const child = spawn("node", [ORTHOS, dictPath, patInPath, patOutPath], {
            stdio: ["pipe", "pipe", "pipe"],
        });

        let stdout = "";
        let stderr = "";
        child.stdout.on("data", (d) => { stdout += d.toString(); });
        child.stderr.on("data", (d) => { stderr += d.toString(); });

        let inputIdx = 0;
        function sendNext() {
            if (inputIdx < inputs.length) {
                child.stdin.write(inputs[inputIdx] + "\n");
                inputIdx++;
                setTimeout(sendNext, 300);
            } else {
                child.stdin.end();
            }
        }
        sendNext();

        const timer = setTimeout(() => {
            child.kill();
            reject(new Error("Timeout — script may be waiting for more input"));
        }, timeout || 30000);

        child.on("close", (code) => {
            clearTimeout(timer);
            resolve({ code, stdout, stderr });
        });
        child.on("error", (err) => {
            clearTimeout(timer);
            reject(err);
        });
    });
}

// ============================================================
// Tests
// ============================================================

let passed = 0;
let failed = 0;

async function test(name, fn) {
    try {
        await fn();
        passed++;
        console.log(`  PASS: ${name}`);
    } catch (e) {
        failed++;
        console.log(`  FAIL: ${name}`);
        console.log(`        ${e.message}`);
    }
}

async function main() {
    setup();

    console.log("Running orthos.js tests...\n");

    // Flow: left_right_hyph_min → hyph_start/finish → pat_start/finish → good/bad/thresh → do_dictionary? (n) → hyphenate? (n)
    const INPUTS = ["1 1", "1 1", "1 1", "1 1 1", "n", "n"];

    await test("Script runs without crashing", async () => {
        const dictFile = writeTmp("test.dic", "ka-ta\nba-cu\nsu-su\n");
        const patIn = writeTmp("in.pat", "");
        const patOut = writeTmp("out.pat", "");
        const { stdout } = await runOrthos(dictFile, patIn, patOut, INPUTS);
        assert.ok(stdout.includes("This is orthos.js"), "Should print banner");
    });

    await test("Output file is created and non-empty", async () => {
        const patOut = path.join(TMP, "out.pat");
        assert.ok(fs.existsSync(patOut), "Output file should exist");
        const content = fs.readFileSync(patOut, "utf-8");
        assert.ok(content.length > 0, "Output should not be empty");
    });

    await test("Reports word count", async () => {
        const dictFile = writeTmp("test2.dic", "a\nab\nabc\nabcd\nabcde\n");
        const patIn = writeTmp("in2.pat", "");
        const patOut = writeTmp("out2.pat", "");
        const { stdout } = await runOrthos(dictFile, patIn, patOut, INPUTS);
        assert.ok(stdout.includes("patterns read in"), "Should report patterns read");
    });

    await test("Output patterns have valid TeX format", async () => {
        const patOut = path.join(TMP, "out.pat");
        const content = fs.readFileSync(patOut, "utf-8");
        const lines = content.trim().split("\n");
        assert.ok(lines.length > 0, "Output should have pattern lines");
        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            // Valid pattern: digits interleaved with letters, e.g. "1k" or "ka1t"
            assert.ok(/^[0-9]*[a-z]+[0-9]*$/.test(trimmed),
                "Pattern line '" + trimmed + "' should be digits+letters");
        }
    });

    await test("Output contains correct good patterns from hyphen boundaries", async () => {
        const patOut = path.join(TMP, "out.pat");
        const content = fs.readFileSync(patOut, "utf-8");
        const lines = content.trim().split("\n");
        // With pat_len=1, pat_dot=0 and good_wt=bad_wt=thresh=1:
        // 't' from ka-ta, 'c' from ba-cu, 's' from su-su are good patterns
        assert.ok(lines.includes("1t"), "Patterns should include '1t' from ka-ta");
        assert.ok(lines.includes("1c"), "Patterns should include '1c' from ba-cu");
        assert.ok(lines.includes("1s"), "Patterns should include '1s' from su-su");
        // Each pattern should appear exactly once
        const countT = lines.filter(l => l === "1t").length;
        const countC = lines.filter(l => l === "1c").length;
        const countS = lines.filter(l => l === "1s").length;
        assert.strictEqual(countT, 1, "'1t' should appear exactly once");
        assert.strictEqual(countC, 1, "'1c' should appear exactly once");
        assert.strictEqual(countS, 1, "'1s' should appear exactly once");
    });

    await test("No arguments exits with code 1", async () => {
        const child = spawn("node", [ORTHOS], { stdio: ["pipe", "pipe", "pipe"] });
        const { code } = await new Promise((resolve) => {
            let stdout = "";
            child.stdout.on("data", (d) => { stdout += d; });
            child.on("close", (code) => resolve({ code, stdout }));
        });
        assert.strictEqual(code, 1, "No-args should exit with code 1");
    });

    await test("Bad dictionary file exits with code 1", async () => {
        const child = spawn("node", [ORTHOS, "/nonexistent/file.dic", "/nonexistent/pat.pat", "out.pat"], { stdio: ["pipe", "pipe", "pipe"] });
        const { code } = await new Promise((resolve) => {
            child.on("close", (code) => resolve({ code }));
        });
        assert.strictEqual(code, 1, "Missing input file should exit with code 1");
    });

    teardown();

    console.log(`\n${passed} passed, ${failed} failed`);
    process.exit(failed > 0 ? 1 : 0);
}

main();
