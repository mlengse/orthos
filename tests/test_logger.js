const { fork } = require("child_process");
const path = require("path");
const assert = require("assert");

const LOGGER_PATH = path.join(__dirname, "..", "logger.js");

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

function runLoggerTest() {
    return new Promise((resolve, reject) => {
        const child = fork(LOGGER_PATH, [], { silent: true });
        let stdout = "";

        child.stdout.on("data", (data) => {
            stdout += data.toString();
        });

        child.send("Hello World!");

        setTimeout(() => {
            child.send("SIGHUP");
        }, 100);

        const timer = setTimeout(() => {
            child.kill();
            reject(new Error("Timeout waiting for logger.js process to exit"));
        }, 5000);

        child.on("exit", (code) => {
            clearTimeout(timer);
            resolve({ code, stdout });
        });

        child.on("error", (err) => {
            clearTimeout(timer);
            reject(err);
        });
    });
}

async function runAllTests() {
    console.log("Running logger.js tests...\n");

    await test("logger.js outputs received messages to stdout and exits cleanly on SIGHUP", async () => {
        const { code, stdout } = await runLoggerTest();
        assert.strictEqual(stdout, "Hello World!", "stdout should match message sent via IPC");
        assert.strictEqual(code, 0, "Process should exit with code 0 on SIGHUP");
    });

    console.log(`\nLogger tests: ${passed} passed, ${failed} failed`);
    if (failed > 0) {
        process.exit(1);
    }
}

if (require.main === module) {
    runAllTests();
}

module.exports = { runAllTests };
