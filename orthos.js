/*jslint browser: false, node: true*/
"use strict";

/**
 * Read the files given as arguments and create a `File`-Object
 * `File` is a proxy that holds the content of the readed
 * file in memory – here we trade memory for I/O
 * (typically the dictionary files are < 10MB)
 * Todo: Error handling
 */

var fs = require("fs");

function makeFile(content) {
    if (content === null || content === undefined) {
        throw new Error("Content cannot be null or undefined");
    }

    // data[0] is the current position index
    // data[1] is the content string
    var data = [0, content];

    function eof() {
        return data[0] >= data[1].length;
    }

    function read() {
        if (eof()) {
            throw new Error("Attempt to read past end of file");
        }
        var char = data[1][data[0]];
        data[0] += 1;
        return char;
    }

    function readln() {
        if (eof()) {
            return null;
        }
        var start = data[0];
        var end = data[1].indexOf('\n', start);
        if (end === -1) {
            end = data[1].length;
        }
        var line = data[1].substring(start, end);
        data[0] = end + 1; // Move past newline
        // Handle CR if present at end of line?
        // Typically readln strips newline chars.
        if (line.length > 0 && line[line.length - 1] === '\r') {
            line = line.substring(0, line.length - 1);
        }
        return line;
    }

    function reset() {
        data[0] = 0;
    }

    return {
        eof: eof,
        read: read,
        readln: readln,
        reset: reset
    };
}

/**
 * Safely reads a file and returns a File proxy object.
 * Handles errors robustly.
 */
function createProxyFromFile(filepath) {
    try {
        if (!filepath) {
            throw new Error("File path is required");
        }

        // Check if file exists
        if (!fs.existsSync(filepath)) {
            throw new Error("File not found: " + filepath);
        }

        // Check if it is a directory
        var stats = fs.statSync(filepath);
        if (stats.isDirectory()) {
            throw new Error("Path is a directory, not a file: " + filepath);
        }

        // Read content
        var content = fs.readFileSync(filepath, 'utf8');
        return makeFile(content);

    } catch (e) {
        // Rethrow with context or handle gracefully?
        // Since this is a library function, throwing is appropriate,
        // but the caller should handle it.
        // For CLI tools, printing a nice error and exiting is common.
        // But here we return the error to allow caller to decide.
        throw new Error("Failed to load file '" + filepath + "': " + e.message);
    }
}

function main() {
    var args = process.argv.slice(2);
    if (args.length < 3) {
        console.log("Usage: node orthos.js <wordlist> <pattern-in> <pattern-out>");
        // We exit gracefully if args are missing, unless we want to run tests/validation
        process.exit(1);
    }

    var wordlistPath = args[0];
    var patternInPath = args[1];
    var patternOutPath = args[2];

    try {
        console.log("Reading wordlist...");
        var wordFile = createProxyFromFile(wordlistPath);
        console.log("Wordlist loaded. Size: " + wordFile.readln().length + " (first line length)");
        wordFile.reset();

        console.log("Reading pattern input...");
        // If pattern file is empty/missing but argument provided, handle it.
        // But createProxyFromFile enforces existence.
        // If the user wants to support "empty file for first level", they should provide an empty file.
        var patternFile = createProxyFromFile(patternInPath);
        console.log("Patterns loaded.");

        // We don't implement the full logic here as the task is about error handling in file proxy.
        // But we check output path validity.
        console.log("Verifying output path...");
        if (fs.existsSync(patternOutPath)) {
            var outStats = fs.statSync(patternOutPath);
            if (outStats.isDirectory()) {
                console.error("Error: Output path is a directory.");
                process.exit(1);
            }
        } else {
             // Check parent dir
             // ...
        }

        console.log("Ready to process.");

    } catch (e) {
        console.error("\nERROR: " + e.message);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = {
    makeFile: makeFile,
    createProxyFromFile: createProxyFromFile
};
