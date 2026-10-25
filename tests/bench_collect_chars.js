/**
 * Benchmark for collectAndSetChars performance
 */

const letter_class = 3;

function origCollectAndSetChars(charsS) {
    const xint = {};
    const xext = [];
    const xclass = {};
    const charsA = [];
    charsS.forEach(function (v) { charsA.push(v); });
    charsA.sort((a, b) => a - b);
    charsA.forEach(function setChar(cc) {
        const c = String.fromCharCode(cc);
        let i = xint[cc];
        if (i === undefined) {
            if (c === c.toUpperCase()) {
                //c is upperCase -> try lowerCase
                i = xint[c.toLowerCase().charCodeAt(0)];
            } else if (c.toUpperCase().length === 1) { //test for length is necessary because "ß".toUpperCase() -> "SS"
                //c ist lowerCase -> try upperCase
                i = xint[c.toUpperCase().charCodeAt(0)];
            }
            if (i === undefined) {
                i = xext.push(c.toLowerCase()) - 1;
                xint[cc] = i;
                xclass[cc] = letter_class;
            } else {
                //other case already exists:
                xint[cc] = i;
                xclass[cc] = letter_class;
            }
        }
    });
    return { xint, xext, xclass };
}

function optCollectAndSetChars(charsS) {
    const xint = {};
    const xext = [];
    const xclass = {};
    const charsA = Array.from(charsS).sort((a, b) => a - b);
    for (let idx = 0; idx < charsA.length; idx++) {
        const cc = charsA[idx];
        let i = xint[cc];
        if (i === undefined) {
            const c = String.fromCharCode(cc);
            const upper = c.toUpperCase();
            if (c === upper) {
                // c is upperCase -> try lowerCase
                const lower = c.toLowerCase();
                i = xint[lower.charCodeAt(0)];
            } else if (upper.length === 1) {
                // c is lowerCase -> try upperCase
                i = xint[upper.charCodeAt(0)];
            }
            if (i === undefined) {
                const lower = c.toLowerCase();
                i = xext.push(lower) - 1;
            }
            xint[cc] = i;
            xclass[cc] = letter_class;
        }
    }
    return { xint, xext, xclass };
}

function runBenchmark() {
    const testSet = new Set();
    for (let cc = 65; cc <= 90; cc++) testSet.add(cc); // A-Z
    for (let cc = 97; cc <= 122; cc++) testSet.add(cc); // a-z
    for (let cc = 192; cc <= 382; cc++) testSet.add(cc); // Extended Latin
    testSet.add(223); // ß

    const iterations = 100000;
    console.log(`Running benchmark with ${iterations} iterations...`);

    // Warmup
    for (let i = 0; i < 1000; i++) {
        origCollectAndSetChars(testSet);
        optCollectAndSetChars(testSet);
    }

    const t0 = process.hrtime.bigint();
    for (let i = 0; i < iterations; i++) {
        origCollectAndSetChars(testSet);
    }
    const t1 = process.hrtime.bigint();
    const origTimeMs = Number(t1 - t0) / 1e6;

    const t2 = process.hrtime.bigint();
    for (let i = 0; i < iterations; i++) {
        optCollectAndSetChars(testSet);
    }
    const t3 = process.hrtime.bigint();
    const optTimeMs = Number(t3 - t2) / 1e6;

    console.log(`Original implementation:  ${origTimeMs.toFixed(3)} ms`);
    console.log(`Optimized implementation: ${optTimeMs.toFixed(3)} ms`);
    console.log(`Improvement: ${((origTimeMs - optTimeMs) / origTimeMs * 100).toFixed(2)}% faster (${(origTimeMs / optTimeMs).toFixed(2)}x)`);
}

if (require.main === module) {
    runBenchmark();
}
