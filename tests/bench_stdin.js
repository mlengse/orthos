/**
 * Benchmark for stdin input parsing performance (shift vs index tracking)
 */

function benchShift(lines) {
    const input = [...lines];
    const start = process.hrtime.bigint();
    let count = 0;
    while (input.length > 0) {
        const line = input.shift();
        if (line) count++;
    }
    const end = process.hrtime.bigint();
    return { timeMs: Number(end - start) / 1e6, count };
}

function benchIndex(lines) {
    const input = [...lines];
    let index = 0;
    const start = process.hrtime.bigint();
    let count = 0;
    while (index < input.length) {
        const line = input[index++];
        if (line) count++;
    }
    const end = process.hrtime.bigint();
    return { timeMs: Number(end - start) / 1e6, count };
}

function runBenchmark() {
    const N = 50000;
    console.log(`Running benchmark with ${N} input lines...`);
    const lines = Array.from({ length: N }, (_, i) => `1 ${i % 10 + 1}`);

    const resShift = benchShift(lines);
    console.log(`shift() approach: ${resShift.timeMs.toFixed(3)} ms`);

    const resIndex = benchIndex(lines);
    console.log(`index tracking approach: ${resIndex.timeMs.toFixed(3)} ms`);

    const speedup = resShift.timeMs / resIndex.timeMs;
    console.log(`Speedup: ${speedup.toFixed(2)}x faster`);
}

if (require.main === module) {
    runBenchmark();
}
