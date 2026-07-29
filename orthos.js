/**
 * orthos.js - Pattern Generator for hyphenation patterns in node.js
 *
 * USAGE
 * (install node version 5.1.0 or newer)
 * node orthos.js <wortliste> <pattern-in> <pattern-out>
 *
 * HISTORY
 * patgen (PATtern GENeration program for the TEX82 hyphenator) was originally
 * written by Frank Liang in PASCAL in 1983 and had revisions in 1991, 1992
 * and 1996 by Peter Breitenloher and in 1996 by Karl Berry. It was originally
 * ported to UNIX by Howard Trickey.
 * patgen is a piece of very well documented and mature software and it is
 * still used to produce new patterns for Frank Liangs hyphenation algorithm.
 *
 * Further reading about patgen and Liangs hyphenation algorithm:
 * - patgen.web: the original patgen
 *   https://www.ctan.org/pkg/patgen
 *   (use weave and tangle to translate the literal programming WEB
 *   to TEX and PASCAL respectively)
 * - Frank  Liang,  Word hy-phen-a-tion by com-puter, STAN-CS-83-977,
 *   Stanford University Ph.D. thesis, 1983
 *   http://tug.org/docs/liang
 *
 * A program called opatgen (with Unicode support and other improvements, by
 * David Antos and Petr Sojka) is also mentioned on http://tug.org/docs/liang
 * put the link is broken and I can't find this software anymore.
 *
 * So why a port of patgen to node.js? (and why JavaScript and not just C?)
 *
 * When patgen was originally written computers were very limited: the PDP-10
 * mainframe computer used by Liang had 256 Kilowords – about 1MB – of memory
 * and could average about 450 kilo instructions per second.
 * Saving memory was critical, characters were encoded in ASCII, students where
 * teached in structured programming using PASCAL.
 *
 * A lot has changed eversince – and many things haven't. Porting patgen to
 * JavaScript/ES6/node.js is aimed at:
 * - learning how patgen works
 * - providing software that runs acceptably fast on current computers
 * - providing a program that can be easily used to produce patterns
 *
 * But why JavaScript?
 * - JavaScript runs on every computer that has at least a browser installed
 * - JavaScript has native Unicode support
 * - It's the language I know best
 *
 * Why the name orthos.js?
 * There's no licence available for patgen. If there would it would most probably
 * be LPPL (https://www.latex-project.org/lppl/). So just to make sure I had to
 * give this another name than "patgen".
 * Then, there is "hydra" (https://github.com/hyphenation/hydra) a Ruby port of
 * patgen. Orthos was the brother of Hydra.
 * And may be some day there will be a hero that kills both…
 *
 * CODING STANDARDS
 * The first version of orthos.js was a more or less 1:1 port from PASCAL to JavaScript.
 * The source is now in a refactoring process where it is step-by-step made
 * more JavaScript-alike.
 * Identifiers copied from the original typically have under_scores in their
 * names, while refactored code uses camelCase.
 * The source is continuously linted by JSLint.
 * The API may change and is not intended to be the same as of the original.
 * Resulting patterns have to be identical to the patterns computed by patgen
 * given the same input (except for their ordering and encoding).
 *
 * ERRORS
 * Please report errors on [githublink]
 *
 * LICENCE
 * Copyright (c) <year> <copyright holders>
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/*jslint browser: false, node: true*/
"use strict";

/**
 * Read the files given as arguments and create a `File`-Object
 * `File` is a proxy that holds the content of the read
 * file in memory – here we trade memory for I/O
 * (typically the dictionary files are < 10MB)
 * Todo: Error handling
 */

const fs = require("fs");
const readline = require("readline");
let stdinInputLines = null;

function makeFile(content) {
    const data = [0, content];

    function eof() {
        return data[0] === data[1].length;
    }

    function reset() {
        data[0] = 0;
    }

    function read() {
        const r = data[1].charAt(data[0]);
        data[0] += 1;
        return r;
    }

    function write(s) {
        data[1] += s;
    }

    function save(path) {
        return new Promise(function (resolve, reject) {
            fs.writeFile(path, data[1], function (err) {
                if (err) {
                    reject(err);
                } else {
                    resolve();
                }
            });
        });
    }

    return {
        data: data,
        eof: eof,
        reset: reset,
        read: read,
        write: write,
        save: save
    };
}

function readFilePromise(path) {
    return new Promise(function (resolve, reject) {
        fs.readFile(path, "utf8", function (err, data) {
            if (err) {
                reject(err);
            } else {
                resolve(data);
            }
        });
    });
}

/**
 * Validate input arguments
 */
function validateInputArgs() {
    if (process.argv.length < 4) {
        println("Usage: node orthos.js <dictionary> <pattern-in> <pattern-out>");
        process.exit(1);
    }
    const dictPath = process.argv[2];
    const patInPath = process.argv[3];
    if (!fs.existsSync(dictPath)) {
        println("Error: Dictionary file not found: " + dictPath);
        process.exit(1);
    }
    if (!fs.statSync(dictPath).isFile()) {
        println("Error: Dictionary path is not a file: " + dictPath);
        process.exit(1);
    }
    const stats = fs.statSync(dictPath);
    if (stats.size > 50 * 1024 * 1024) {
        println("Error: Dictionary file exceeds 50 MB size limit: " + dictPath);
        process.exit(1);
    }
}

/**
 * Get promises for input files
 */
const dictionaryProm = readFilePromise(process.argv[2]);
const patternInProm = readFilePromise(process.argv[3]);

/**
 * Create output files
 */
const patout = makeFile("");

/**
 * variables for fullfilled file promises
 */
let dictionary;
let patterns;


/**
 * Globally defined const and var/let
 * Todo:
 * [] Comment each var/const/let
 * [] Check which vars/consts could be made local or eliminated
 * [] Check if some var/let can be redefined as const
 * [] use let instead of var (beware no-opt of compound let statements)
 */

/**
 * First line printed out when orthos.js is run
 */
const banner = "This is orthos.js for node.js, Version 1.0b";

/**
 * space for pattern trie
 * originally this was 55000, but that's to small for the german dictionary
 */
const trie_size = 55000 * 40;  // Reduced from 55000*80 (43 MB → 22 MB)

/**
 * space for pattern count trie, must be less than trie_size and greater
 * than the number of occurrences of any pattern in the dictionary
 * originally this was 26000, but that's to small for the german dictionary
 */
const triec_size = 32000 * 80;

/**
 * size of output hash table, should be a multiple of 510 (= 2*255)
 */
const max_ops = 510 * 10;

/**
 * maximum number of levels+1, also used to denote bad patterns
 */
const max_val = 10;

/**
 * maximum pattern length,
 * also maximum length of external representation of a 'letter'
 */
const max_dot = 15;

/**
 * maximum word length,
 */
const max_len = 50;

/**
 * maximum length of input lines, at least max_len
 * originaly this was const set to 80, but it's faster
 * to dynamically assign the buf-array and push new values.
 */
let buf_len;

/**
 * The following vars are the parameters for orthos.js
 * They are prompted to the user for each run.
 */
let pat_start;
let pat_finish;
let hyph_start;
let hyph_finish;
let good_wt;
let bad_wt;
let thresh;

/**
 * For the purpose of this program (see trie creation) each character
 * is mapped to an internal code of type int without any relations to
 * some encoding. The characters '0'...'9' and '.' have special meanings
 * and are used in every language and thus always mapped to the same int:
 * '0'->0 ... '9'->9 and '.'->10 (see function initialize())
 * '.' marks the beginning an end of a word. A pattern like '.ab1' only matches
 * at the beginning of a word that begins with 'ab' while a pattern like 'ab1'
 * matches at every position in a word.
 */
let edge_of_word;

/**
 * When reading the dictionary and patterns the characters have to be
 * recognized as fast as possible. Reading the charCode and then decide
 * what kind of character it is takes to much time.
 * Thus we traverse the dictionary at the beginning and collect all characters,
 * map them to a internalCharCode and assign them a class.
 *
 * xint: charCode -> internalCharCode
 * xext: internalCharCode -> char
 * xdig: value -> digit
 * xclass: char -> one of the following classes:
 *
 * digit_class: chars '0'...'9' (hyphen values or weights)
 * hyf_class: chars '.', '-' and '*' (bad, missed, good hyphens)
 * letter_class: chars representing a letter
 * invalid_class: chars that should not occur
 *
 * The only char that changes its class is '.':
 * reading dictionary: invalid_class
 * reading patterns: letter_class
 * writing pattmp: hyf_class
 */
const digit_class = 1;
const hyf_class = 2;
const letter_class = 3;
const no_hyf = 10;// ''
const err_hyf = 11;// '#'
const is_hyf = 12;// '-'
const found_hyf = 13;// '*'

const xclass = {};
const xint = {};
let xdig = [];
const xext = [];

/**
 * cmin...cmax is the range of internalCharCodes (cmax = xext.length - 1)
 * We start at 1 since 0:'0' will never be used in dictionary or pattern
 * cnum is the number of internalCharCodes (cnum = xext.length)
 */
const cmin = 1;
let cmax;
let cnum;

/**
 * The trie is stored in separate typed arrays:
 * trie_* (pattern trie)
 * triec_* (pattern count trie)
 * *_c (char as internalCharCode)
 * *_l (link resp good count)
 * *_r (back resp bad count or output)
 * *_taken (1: triebase is taken, 0: triebase is free)
 * ops: outputs (Hash-object with three properties)
 */
const trie_c = new Uint32Array(trie_size);
const trie_l = new Uint32Array(trie_size);
const trie_r = new Uint32Array(trie_size);
const trie_taken = new Uint8Array(trie_size);
const triec_c = new Uint32Array(triec_size);
const triec_l = new Uint32Array(triec_size);
const triec_r = new Uint32Array(triec_size);
const triec_taken = new Uint8Array(triec_size);
const ops = [];

/**
 * When some trie state is being worked on, an unpacked version of the state
 * is kept in position 1..qmax of trieq_*
 * qmax_thresh controls the density of first-fit packing (0: sparse)
 * Defaults to 5: After deleting outputs this is set to 7
 * because the trie is sparser
 * Todo: investigate this. Maybe space could be treated for speed
 */
const trieq_c = new Uint32Array(50);
const trieq_l = new Uint32Array(50);
const trieq_r = new Uint32Array(50);
let qmax;
let qmax_thresh;

let trie_max;//maximum occupied trie node
let trie_bmax;//maximum base of trie family
let trie_count;//number of occupied trie nodes, for space usage statistics
let op_count;//number of outpust in hash table


/**
 * The trie_root is on position 1. A link to 0 marks the end of a pattern
 */
const trie_root = 1;
const triec_root = 1;

/**
 * pat is the current pattern of length pat_len
 */
const pat = new Uint32Array(50);
let pat_len;

let triec_max;//maximum occupied trie node
let triec_bmax;//maximum base of trie family
let triec_count;//number of occupied trie nodes, for space usage statistics
let triec_kmax;//shows growth of trie during pass
let pat_count;//number of patterns in count trie

const word = new Uint8Array(max_len);//current word
const dots = new Uint8Array(max_len);//current hyphens
const dotw = new Uint8Array(max_len);//dot weights
const hval = new Uint8Array(max_len);//hyphenation values
const no_more = new Uint8Array(max_len);//positions 'knocked out'
let wlen;//length of current word
let word_wt;//the global word weight
let wt_chg;//indicates word_wt has changed

//block 55
let left_hyphen_min;//minimal length of string before first hyphenatiom
let right_hyphen_min;//minimal length of string after last hyphenatiom

//block 66
let good_pat_count;//number of good patterns added at end of pass
let bad_pat_count;//number of bad patterns added at end of pass
let good_count;//good hyphen count
let bad_count;//bad hyphen count
let miss_count;//missed hyphen coung
let level_pattern_count;//number of good patterns at level
let more_to_come;//set to true if the quality of a pattern is ambiguous

/**
 * If hyphp is set to true, do_dictionary will write out a copy of the
 * dictionary as hyphenated by the current set of patterns. If procesp is
 * set to true, do_dictionary will collect pattern statistics for patterns
 * with length pat_len and hyphen position pat_dot, at level hyph_level.
 */
let procesp;
let hyphp;
let pat_dot;
let hyph_level;
let filnam = "";

let hyf_min;//left_hyphen_min + 1
let hyf_max;//right_hyphen_min + 1
let hyf_len;//hyf_min + hyf_max

let good_dot;//is_hyf in odd (collecting) passes, err_hyf in even passes
let bad_dot;//no_hyf in odd (collecting) passes, found_hyf in even passes
let dot_min;//analoguos to hyf_min
let dot_max;//analoguos to hyf_max
let dot_len;//analoguos to hyf_len

//block 91
let max_pat;//largest hyphenation value found in any pattern

/**
 * Some helper functions
 */
const cp = require("child_process");
const logger = cp.fork(require("path").resolve(__dirname, "logger.js"));
process.on("exit", function () { logger.kill(); });
process.on("SIGINT", function () { logger.kill(); process.exit(1); });
process.on("SIGTERM", function () { logger.kill(); process.exit(1); });

function print(s) {
    process.stdout.write(s);
}

function println(s) {
    process.stdout.write(s + "\n");
}

validateInputArgs();

//block 10
function error(msg) {
    println(msg);
    process.exit(1);

}
function overflow(msg1) {
    error("PATGEN capacity exceeded, sorry [" + msg1 + "].");
}

/**
 * initialize() sets up xint, xext and xclass with characters that are language independent
 */
function initialize() {
    //setup internal representation for digits
    String("0123456789").split("").forEach(function (c, i) {
        const cc = c.charCodeAt(0);
        xext[i] = String.fromCharCode(cc);
        xint[cc] = i;
        xclass[cc] = digit_class;
    });
    //setup IR for ".": the edge_of_word marker (can occur in pattern files)
    xext[10] = ".";
    xint[46] = 10;
    xclass[46] = letter_class;
    edge_of_word = 10;

    //setup IR for "#": the err_hyf marker
    xext[11] = "#";
    xint[35] = 11;
    xclass[35] = hyf_class;

    //setup IR for "-": the is_hyf marker (occurs in dictionary)
    xext[12] = "-";
    xint[45] = 12;
    xclass[45] = hyf_class;

    //setup IR for "*": the found_hyf marker
    xext[13] = "*";
    xint[42] = 13;
    xclass[42] = hyf_class;
}



//block 35
function first_fit() {
    let s;
    let t;
    let q;
    let doSearchLoop = true;
    let continueLoop = false;
    //block 36
    t = (qmax > qmax_thresh)
        ? trie_r[trie_max + 1]
        : 0;

    while (doSearchLoop) {
        continueLoop = false;
        t = trie_l[t];
        s = t - trieq_c[1];
        //block 37
        if (s > trie_size - cnum) {
            overflow(trie_size + " pattern trie nodes");
        }
        while (trie_bmax < s) {
            trie_bmax += 1;
            trie_taken[trie_bmax] = 0;
            trie_c[trie_bmax + cmax] = 0;
            trie_l[trie_bmax + cmax] = trie_bmax + cnum;
            trie_r[trie_bmax + cnum] = trie_bmax + cmax;
        }
        // end block 37
        if (trie_taken[s] === 0) {
            q = qmax;
            while (q >= 2) {
                if (trie_c[s + trieq_c[q]] !== 0) {
                    continueLoop = true;
                }
                q -= 1;
            }
        } else {
            continueLoop = true;
        }
        doSearchLoop = continueLoop;
    }
    //end block 36
    q = 1;
    while (q <= qmax) {
        t = s + trieq_c[q];
        trie_l[trie_r[t]] = trie_l[t];
        trie_r[trie_l[t]] = trie_r[t];
        trie_c[t] = trieq_c[q];
        trie_l[t] = trieq_l[q];
        trie_r[t] = trieq_r[q];
        if (t > trie_max) {
            trie_max = t;
        }
        q += 1;
    }
    trie_taken[s] = 1;
    return s;
}

//block 38
function unpack(s) {
    let c;
    let t;
    qmax = 1;
    c = cmin;
    while (c <= cmax) {
        t = s + c;
        if (trie_c[t] === c) {
            trieq_c[qmax] = c;
            trieq_l[qmax] = trie_l[t];
            trieq_r[qmax] = trie_r[t];
            qmax += 1;
            trie_r[trie_l[0]] = t;
            trie_l[t] = trie_l[0];
            trie_l[0] = t;
            trie_r[t] = 0;
            trie_c[t] = 0;
        }
        c += 1;
    }
    trie_taken[s] = 0;
}
//block 34
function init_pattern_trie() {
    let c = 0; //internal_code
    let h = 1; //opt_type
    while (c <= cmax) {
        trie_c[trie_root + c] = c;
        trie_l[trie_root + c] = 0;
        trie_r[trie_root + c] = 0;
        trie_taken[trie_root + c] = 0;
        c += 1;
    }
    trie_taken[trie_root] = 1;
    trie_bmax = trie_root;
    trie_max = trie_root + cmax;
    trie_count = cnum;
    qmax_thresh = 5;
    trie_l[0] = trie_max + 1;
    trie_r[trie_max + 1] = 0;
    while (h <= max_ops) {
        ops[h] = {val: 0, dot: 0, op: 0};
        h += 1;
    }
    op_count = 0;
}

//block 39
function new_trie_op(v, d, n) {
    let h;
    h = ((n + 313 * d + 361 * v) % max_ops) + 1;
    while (true) {
        if (ops[h].val === 0) {
            op_count += 1;
            if (op_count === max_ops) {
                overflow(max_ops + " outputs");
            }
            ops[h].val = v;
            ops[h].dot = d;
            ops[h].op = n;
            return h;
        }
        if (ops[h].val === v && ops[h].dot === d && ops[h].op === n) {
            return h;
        }
        if (h > 1) {
            h -= 1;
        } else {
            h = max_ops;
        }
    }
}


//block 41
function insert_pattern(val, dot) {
    let i;
    let s;
    let t;
    i = 1;
    s = trie_root + pat[i];
    t = trie_l[s];
    while (t > 0 && i < pat_len) {
        i += 1;
        t += pat[i];
        if (trie_c[t] !== pat[i]) {
            //begin block 42
            if (trie_c[t] === 0) {
                trie_l[trie_r[t]] = trie_l[t];
                trie_r[trie_l[t]] = trie_r[t];
                trie_c[t] = pat[i];
                trie_l[t] = 0;
                trie_r[t] = 0;
                if (t > trie_max) {
                    trie_max = t;
                }
            } else {
                unpack(t - pat[i]);
                trieq_c[qmax] = pat[i];
                trieq_l[qmax] = 0;
                trieq_r[qmax] = 0;
                t = first_fit();
                trie_l[s] = t;
                t += pat[i];
            }
            trie_count += 1;
            //end block 42
        }
        s = t;
        t = trie_l[s];
    }
    trieq_l[1] = 0;
    trieq_r[1] = 0;
    qmax = 1;
    while (i < pat_len) {
        i += 1;
        trieq_c[1] = pat[i];
        t = first_fit();
        trie_l[s] = t;
        s = t + pat[i];
        trie_count += 1;
    }
    trie_r[s] = new_trie_op(val, dot, trie_r[s]);
}



function init_count_trie() {
    let c = 0;
    while (c <= cmax) {
        triec_c[triec_root + c] = c;
        triec_l[triec_root + c] = 0;
        triec_r[triec_root + c] = 0;
        triec_taken[triec_root + c] = 0;
        c += 1;
    }
    triec_taken[triec_root] = 1;
    triec_bmax = triec_root;
    triec_max = triec_root + cmax;
    triec_count = cnum;
    triec_kmax = 4096;
    triec_l[0] = triec_max + 1;
    triec_r[triec_max + 1] = 0;
    pat_count = 0;
}

//block 45
function firstc_fit() {
    let a;
    let b;
    let q;
    let doSearchLoop = true;
    let continueLoop = false;
    //begin block 46
    if (qmax > 3) {
        a = triec_r[triec_max + 1];
    } else {
        a = 0;
    }
    while (doSearchLoop) {
        continueLoop = false;
        a = triec_l[a];
        b = a - trieq_c[1];
        //begin block 47
        if (b > (triec_kmax - cnum)) {
            if (triec_kmax === triec_size) {
                overflow(triec_size + " count trie nodes");
            }
            print(~~(triec_kmax / 1024) + "K ");
            if (triec_kmax > (triec_size - 4096)) {
                triec_kmax = triec_size;
            } else {
                triec_kmax += 4096;
            }
        }
        while (triec_bmax < b) {
            triec_bmax += 1;
            triec_taken[triec_bmax] = 0;
            triec_c[triec_bmax + cmax] = 0;
            triec_l[triec_bmax + cmax] = triec_bmax + cnum;
            triec_r[triec_bmax + cnum] = triec_bmax + cmax;
        }
        //end block 47
        if (triec_taken[b] === 0) {
            q = qmax;
            while (q >= 2) {
                if (triec_c[b + trieq_c[q]] !== 0) {
                    continueLoop = true;
                }
                q -= 1;
            }
        } else {
            continueLoop = true;
        }
        doSearchLoop = continueLoop;
    }
    //end block 46
    q = 1;
    while (q <= qmax) {
        a = b + trieq_c[q];
        triec_l[triec_r[a]] = triec_l[a];
        triec_r[triec_l[a]] = triec_r[a];
        triec_c[a] = trieq_c[q];
        triec_l[a] = trieq_l[q];
        triec_r[a] = trieq_r[q];
        if (a > triec_max) {
            triec_max = a;
        }
        q += 1;
    }
    triec_taken[b] = 1;
    return b;
}

//block 48
function unpackc(b) {
    let c = cmin;
    let a;
    qmax = 1;
    while (c <= cmax) {
        a = b + c;
        if (triec_c[a] === c) {
            trieq_c[qmax] = c;
            trieq_l[qmax] = triec_l[a];
            trieq_r[qmax] = triec_r[a];
            qmax += 1;
            triec_r[triec_l[0]] = a;
            triec_l[a] = triec_l[0];
            triec_l[0] = a;
            triec_r[a] = 0;
            triec_c[a] = 0;
        }
        c += 1;
    }
    triec_taken[b] = 0;
}



//block 49
function insertc_pat(fpos) {
    let spos;
    let a;
    let b;
    spos = fpos - pat_len;
    spos += 1;
    b = triec_root + word[spos];
    a = triec_l[b];
    while (a > 0 && spos < fpos) {
        spos += 1;
        a += word[spos];
        if (triec_c[a] !== word[spos]) {
            //begin block 50
            if (triec_c[a] === 0) {
                triec_l[triec_r[a]] = triec_l[a];
                triec_r[triec_l[a]] = triec_r[a];
                triec_c[a] = word[spos];
                triec_l[a] = 0;
                triec_r[a] = 0;
                if (a > triec_max) {
                    triec_max = a;
                }
            } else {
                unpackc(a - word[spos]);
                trieq_c[qmax] = word[spos];
                trieq_l[qmax] = 0;
                trieq_r[qmax] = 0;
                a = firstc_fit();
                triec_l[b] = a;
                a += word[spos];
            }
            triec_count += 1;
            //end block 50
        }
        b = a;
        a = triec_l[a];
    }
    trieq_l[1] = 0;
    trieq_r[1] = 0;
    qmax = 1;
    while (spos < fpos) {
        spos += 1;
        trieq_c[1] = word[spos];
        a = firstc_fit();
        triec_l[b] = a;
        b = a + word[spos];
        triec_count += 1;
    }
    pat_count += 1;
    return b;
}

//block 51
const pattmp = makeFile("");

//block 52
let buf = [];
let buf_ptr;

//block 53
function print_buf() {
    println(buf.join(""));
}

function bad_input(msg) {
    print_buf();
    error(msg);
}

//block 52
function read_buf(file) {
    var nl = file.data[1].indexOf("\n", file.data[0]);
    if (nl === -1) {
        buf = file.data[1].slice(file.data[0]).split("");
        file.data[0] = file.data[1].length;
    } else {
        buf = file.data[1].slice(file.data[0], nl).split("");
        file.data[0] = nl + 1;
    }
    buf_len = buf.length;
}




//block 61
function find_letters(b, i) {
    let c = cmin;
    let a;
    let j;
    let l;
    if (i === 0) {
        init_count_trie();
    }
    while (c <= cmax) {
        a = b + c;
        if (trie_c[a] === c) {
            pat[i] = c;
            if (trie_r[a] === 0) {
                find_letters(trie_l[a], i + 1);
            } else if (trie_l[a] === 0) {
                //begin block 62
                l = triec_root + trie_r[a];
                j = 1;
                while (j <= i - 1) {
                    if (triec_max === triec_size) {
                        overflow(triec_size + " count trie nodes (fl)");
                    }
                    triec_max += 1;
                    triec_l[l] = triec_max;
                    l = triec_max;
                    triec_c[l] = pat[j];
                    j += 1;
                }
                triec_l[l] = 0;
                //end block 62
            }
        }
        c += 1;
    }
}



//block 64
function traverse_count_trie(b, i) {
    let c = cmin;
    let a;
    while (c <= cmax) {
        a = b + c;
        if (triec_c[a] === c) {
            pat[i] = c;
            if (i < pat_len) {
                traverse_count_trie(triec_l[a], i + 1);
            } else {
                //begin block 65
                if ((good_wt * triec_l[a]) < thresh) { //hopeless pattern
                    //println("HOPELESS:", pat.map(v => xext[v]), triec_l[a], triec_r[a]);
                    insert_pattern(max_val, pat_dot);
                    bad_pat_count += 1;
                } else if ((good_wt * triec_l[a] - bad_wt * triec_r[a]) >= thresh) { //good pattern
                    //println(`INSERT: ${pat.map(v => xext[v])}, ${triec_l[a]}, ${triec_r[a]}`);
                    insert_pattern(hyph_level, pat_dot);
                    good_pat_count += 1;
                    good_count += triec_l[a];
                    bad_count += triec_r[a];
                } else {
                    //println("MORE2COME:", pat.map(v => xext[v]), triec_l[a], triec_r[a]); //can't decide yet
                    more_to_come = true;
                }
                //end block 65
            }
        }
        c += 1;
    }
}

//block 67
function collect_count_trie() {
    good_pat_count = 0;
    bad_pat_count = 0;
    good_count = 0;
    bad_count = 0;
    more_to_come = false;
    traverse_count_trie(triec_root, 1);
    print(good_pat_count + " good and " + bad_pat_count + " bad patterns added");
    level_pattern_count += good_pat_count;
    if (more_to_come) {
        println(" (more to come)");
    } else {
        println(" ");
    }
    print("finding " + good_count + " good and " + bad_count + " bad hyphens");
    if (good_pat_count > 0) {
        println(" efficiency = " + Number(good_count / (good_pat_count + bad_count / (thresh / good_wt))).toFixed(2));
    } else {
        println(" ");
    }
    println("pattern trie has " + trie_count + " nodes, trie_max = " + trie_max + ", " + op_count + " outputs");
}

//block 68
function delete_patterns(s) {
    let c = cmin;
    let t;
    let all_freed;
    let h;
    let n;
    all_freed = true;
    while (c <= cmax) {
        t = s + c;
        if (trie_c[t] === c) {
            //begin block 69
            h = 0;
            ops[0] = {op: trie_r[t]};
            n = ops[0].op;
            while (n > 0) {
                if (ops[n].val === max_val) {
                    ops[h].op = ops[n].op;
                } else {
                    h = n;
                }
                n = ops[h].op;
            }
            trie_r[t] = ops[0].op;
            //end block 69
            if (trie_l[t] > 0) {
                trie_l[t] = delete_patterns(trie_l[t]);
            }
            if (trie_l[t] > 0 || trie_r[t] > 0 || s === trie_root) {
                all_freed = false;
            } else {
                //begin block 70
                trie_l[trie_r[trie_max + 1]] = t;
                trie_r[t] = trie_r[trie_max + 1];
                trie_l[t] = trie_max + 1;
                trie_r[trie_max + 1] = t;
                trie_c[t] = 0;
                trie_count -= 1;
                //end block 70
            }
        }
        c += 1;
    }
    if (all_freed) {
        trie_taken[s] = 0;
        s = 0;
    }
    return s;
}

//block 71
function delete_bad_patterns() {
    let old_op_count;
    let old_trie_count;
    let h = 1;
    old_op_count = op_count;
    old_trie_count = trie_count;
    delete_patterns(trie_root);
    while (h <= max_ops) {
        if (ops[h].val === max_val) {
            ops[h].val = 0;
            op_count -= 1;
        }
        h += 1;
    }
    println((old_trie_count - trie_count) + " nodes and " + (old_op_count - op_count) + " outputs deleted");
    qmax_thresh = 7;
}

//block 72
function output_patterns(s, pat_len, indent) {
    indent = indent || 0;
    let c = cmin;
    let t;
    let h;
    let d;
    while (c <= cmax) {
        t = s + c;
        if (trie_c[t] === c) {
            pat[pat_len] = c;
            h = trie_r[t];
            if (h > 0) {
                //begin block 73
                d = 0;
                while (d <= pat_len) {
                    hval[d] = 0;
                    d += 1;
                }
                do {
                    d = ops[h].dot;
                    if (hval[d] < ops[h].val) {
                        hval[d] = ops[h].val;
                    }
                    h = ops[h].op;
                } while (h !== 0);
                if (hval[0] > 0) {
                    patout.write(xdig[hval[0]]);
                }
                d = 1;
                while (d <= pat_len) {
                    patout.write(xext[pat[d]]);
                    if (hval[d] > 0) {
                        patout.write(xdig[hval[d]]);
                    }
                    d += 1;
                }
                patout.write("\n");
                 //end block 73
            }
            if (trie_l[t] > 0) {
                output_patterns(trie_l[t], pat_len + 1, indent + 4);
            }
        }
        c += 1;
    }
}


//block 76
function read_word() {
    let cc = dictionary.data[1].charCodeAt(dictionary.data[0]);
    word[1] = edge_of_word;
    wlen = 1;
    while (cc !== 10) {
        switch (xclass[cc]) {
        case letter_class:
            wlen += 1;
            word[wlen] = xint[cc];
            dots[wlen] = no_hyf;
            dotw[wlen] = word_wt;
            break;
        case hyf_class:
            dots[wlen] = xint[cc];
            break;
        case digit_class:
            if (wlen === 1) {
                if (xint[cc] !== word_wt) {
                    wt_chg = true;
                    word_wt = xint[cc];
                }
            } else {
                dotw[wlen] = xint[cc];
            }
            break;
        default:
            bad_input("Bad character");
        }
        dictionary.data[0] += 1;
        cc = dictionary.data[1].charCodeAt(dictionary.data[0]);
    }
    wlen += 1;
    word[wlen] = edge_of_word;
    dictionary.data[0] += 1;
}

//block 77
function hyphenate() {
    let spos = wlen - hyf_max;
    let dpos = 0;
    let fpos = 0;
    let t;
    let h;
    let v;

    while (spos >= 0) {
        no_more[spos] = 0;
        hval[spos] = 0;
        fpos = spos + 1;
        t = trie_root + word[fpos];
        while (trie_c[t] === word[fpos]) {
            h = trie_r[t];
            while (h > 0) {
                //begin block 80
                dpos = spos + ops[h].dot;
                v = ops[h].val;
                if (v < max_val && hval[dpos] < v) {
                    hval[dpos] = v;
                }
                if (v >= hyph_level && (fpos - pat_len) <= (dpos - pat_dot) && (dpos - pat_dot) <= spos) {
                    no_more[dpos] = 1;
                }
                h = ops[h].op;
                //end block 80
            }
            t = trie_l[t];
            if (t === 0) {
                break;
            }
            fpos += 1;
            t += word[fpos];
        }
        spos -= 1;
    }
}

//block 81
function change_dots() {
    let dpos = wlen - hyf_max;
    while (dpos >= hyf_min) {
        if ((hval[dpos] % 2) === 1) {
            dots[dpos] += 1;
        }
        switch (dots[dpos]) {
        case found_hyf:
            good_count += dotw[dpos];
            break;
        case err_hyf:
            bad_count += dotw[dpos];
            break;
        case is_hyf:
            miss_count += dotw[dpos];
            break;
        }
        dpos -= 1;
    }
}


//bloch 82
function output_hyphenated_word() {
    let dpos = 2;
    if (wt_chg) {
        pattmp.write(xdig[word_wt]);
        wt_chg = false;
    }
    while (dpos <= wlen - 2) {
        pattmp.write(xext[word[dpos]]);
        if (dots[dpos] !== no_hyf) {
            pattmp.write(xext[dots[dpos]]);
        }
        if (dotw[dpos] !== word_wt) {
            pattmp.write(xdig[dotw[dpos]]);
        }
        dpos += 1;
    }
    pattmp.write(xext[word[wlen - 1]] + "\n");
}


//block 83
/*
For each dot position in the current word, the do word routine first checks to see if we need to consider it.
It might be knocked out or a dot we don’t care about. That is, when considering hyphenating patterns,
for example, we don’t need to count hyphens already found. If a relevant dot is found, we increment the count
in the count trie for the corresponding pattern, inserting it first if necessary.
At this point of the program range check violations may occur if these counts are incremented beyond triec max;
it would, however, be too expensive to prevent this.
*/
function do_word() {
    let spos;
    let dpos = wlen - dot_max;
    let fpos;
    let a;
    while (dpos >= dot_min) {
        spos = dpos - pat_dot;
        fpos = spos + pat_len;
        //begin block 86
        if (no_more[dpos] === 0) {
            if (dots[dpos] === good_dot) {
                spos += 1;
                a = triec_root + word[spos];
                while (spos < fpos) {
                    spos += 1;
                    a = triec_l[a] + word[spos];
                    if (triec_c[a] !== word[spos]) {
                        a = insertc_pat(fpos);
                        break;
                    }
                }
                triec_l[a] += dotw[dpos];
            } else if (dots[dpos] === bad_dot) {
                spos += 1;
                a = triec_root + word[spos];
                while (spos < fpos) {
                    spos += 1;
                    a = triec_l[a] + word[spos];
                    if (triec_c[a] !== word[spos]) {
                        a = insertc_pat(fpos);
                        break;
                    }
                }
                triec_r[a] += dotw[dpos];
            }
            //end block 86
        }
        dpos -= 1;
    }
}

//block 88
function do_dictionary() {
    good_count = 0;
    bad_count = 0;
    miss_count = 0;
    word_wt = 1;
    wt_chg = false;
    //begin block 79
    hyf_min = left_hyphen_min + 1;
    hyf_max = right_hyphen_min + 1;
    hyf_len = hyf_min + hyf_max;
    //end block 79
    if (procesp) {
        dot_min = pat_dot;
        dot_max = pat_len - pat_dot;
        if (dot_min < hyf_min) {
            dot_min = hyf_min;
        }
        if (dot_max < hyf_max) {
            dot_max = hyf_max;
        }
        dot_len = dot_min + dot_max;
        if ((hyph_level % 2) === 1) { //is odd
            good_dot = is_hyf;
            bad_dot = no_hyf;
        } else {
            good_dot = err_hyf;
            bad_dot = found_hyf;
        }
        init_count_trie();
        println("processing dictionary with pat_len " + pat_len + ", pat_dot = " + pat_dot);
    }
    if (hyphp) {
        filnam = "pattmp." + (hyph_level + 1);
        println("writing " + filnam);
    }
    //begin block 89
    dictionary.reset();
    while (!dictionary.eof()) {
        read_word(); //see block 76
        if (wlen >= hyf_len) {
            hyphenate();
            change_dots();
        }
        if (hyphp) {
            if (wlen > 2) {
                output_hyphenated_word();
            }
        }
        if (procesp) {
            if (wlen >= dot_len) {
                do_word();
            }
        }
    }
    //end block 89
    println(" ");
    println(Number(good_count).toString() + " good, " + Number(bad_count).toString() + " bad, " + Number(miss_count).toString() + " missed");
    if ((good_count + miss_count) > 0) {
        println(Number(100 * good_count / (good_count + miss_count)).toFixed(2) + " %, " + Number(100 * bad_count / (good_count + miss_count)).toFixed(2) + " %, " + Number(100 * miss_count / (good_count + miss_count)).toFixed(2) + " %");
    }
    if (procesp) {
        println(pat_count + " patterns, " + triec_count + " nodes in count trie, " + "triec_max = " + triec_max);
    }
    if (hyphp) {
        pattmp.save(filnam).catch(
            function (err) {
                println(err);
            }
        );
    }
}



//block 90
function read_patterns() {
    let cc;
    let d;
    let i;
    level_pattern_count = 0;
    max_pat = 0;
    patterns.reset();
    while (!patterns.eof()) {
        read_buf(patterns);
        level_pattern_count += 1;
        //begin block 92
        pat_len = 0;
        buf_ptr = 0;
        hval[0] = 0;
        while (buf_ptr < buf_len) {
            cc = buf[buf_ptr].charCodeAt(0);
            switch (xclass[cc]) {
            case digit_class:
                d = xint[cc];
                if (d >= max_val) {
                    bad_input("Bad hyphenation value");
                }
                if (d > max_pat) {
                    max_pat = d;
                }
                hval[pat_len] = d;
                break;
            case letter_class:
                pat_len += 1;
                if (pat_len >= 50) {
                    bad_input("Pattern exceeds maximum length");
                }
                hval[pat_len] = 0;
                pat[pat_len] = xint[cc];
                break;
            default:
                bad_input("Bad character: ", String.fromCharCode(cc));
            }
            buf_ptr += 1;
        }
        //end block 92
        //block 93
        if (pat_len > 0) {
            i = 0;
            while (i <= pat_len) {
                if (hval[i] !== 0) {
                    insert_pattern(hval[i], i);
                }
                if (i > 1) {
                    if (i < pat_len) {
                        if (pat[i] === edge_of_word) {
                            bad_input("Bad edge_of_word");
                        }
                    }
                }
                i += 1;
            }
        }
    }
    //r.close(patterns);
    println(level_pattern_count + " patterns read in");
    println("pattern trie has " + trie_count + " nodes, trie_max = " + trie_max + ", " + op_count + " outputs");
}


//block 94
function generateLevel() {
    //block 95
    let j;
    let k = 0;
    let dot1;
    const more_this_level = [];
    //generate a level 96
    while (k <= max_dot) {
        more_this_level[k] = true;
        k += 1;
    }
    j = pat_start;
    while (j <= pat_finish) {
        pat_len = j;
        pat_dot = ~~(pat_len / 2); //integer division
        dot1 = pat_dot * 2;
        do {
            pat_dot = dot1 - pat_dot;
            dot1 = pat_len * 2 - dot1 - 1;
            if (more_this_level[pat_dot]) {
                do_dictionary();
                collect_count_trie();
                more_this_level[pat_dot] = more_to_come;
            }
        } while (pat_dot !== pat_len);
        k = max_dot;
        while (k >= 1) {
            if (!more_this_level[k - 1]) {
                more_this_level[k] = false;
            }
            k -= 1;
        }
        j += 1;
    }
    //end block 96
    delete_bad_patterns();
    println("total of " + level_pattern_count + " patterns at hyph_level " + hyph_level);
}

function askDoDictionary() {
    if (stdinInputLines) {
        const chunk = (stdinInputLines.shift() || "n").trim();
        if (chunk === "y" || chunk === "Y") {
            do_dictionary();
            logger.send("SIGHUP");
        } else {
            logger.send("SIGHUP");
            process.exit(0);
        }
        return;
    }
    const rl = readline.createInterface({input: process.stdin, output: process.stdout});
    let responded = false;
    rl.setPrompt("hyphenate word list? ");
    rl.prompt();

    rl.on("line", function (line) {
        if (responded) { return; }
        responded = true;
        const chunk = line.trim();
        rl.close();
        if (chunk === "y" || chunk === "Y") {
            do_dictionary();
            logger.send("SIGHUP");
        } else {
            logger.send("SIGHUP");
            process.exit(0);
        }
    });
}


function doLevels(currLevel) {
    hyph_level = max_pat;
    if (currLevel <= hyph_finish) {
        hyph_level = currLevel;
        level_pattern_count = 0;
        if (hyph_level > hyph_start) {
            println(" ");
        } else if (hyph_start <= max_pat) {
            println("Largest hyphenation value " + max_pat + " in patterns should be less than hyph_start");
        }
        getPat(hyph_level);
    } else {
        find_letters(trie_l[trie_root], 1);
        output_patterns(trie_root, 1);
        try {
            fs.writeFileSync(process.argv[4], patout.data[1], "utf-8");
        } catch (err) {
            println(err);
        }
        //block 97
        procesp = false;
        hyphp = true;
        askDoDictionary();
    }
}

//var profiler = require("v8-profiler");
function getGBT(currLevel) {
    if (stdinInputLines) {
        let n1 = undefined, n2 = undefined, n3 = undefined;
        while (stdinInputLines.length > 0) {
            const chunks = stdinInputLines.shift().trim().split(" ");
            if (chunks.length >= 3) {
                n1 = parseInt(chunks[0], 10);
                n2 = parseInt(chunks[1], 10);
                n3 = parseInt(chunks[2], 10);
            } else if (chunks.length === 1) {
                if (n1 === undefined) { n1 = parseInt(chunks[0], 10); continue; }
                else if (n2 === undefined) { n2 = parseInt(chunks[0], 10); continue; }
                else { n3 = parseInt(chunks[0], 10); }
            } else { continue; }
            if (n1 >= 1 && n2 >= 1 && n3 >= 1) {
                good_wt = n1; bad_wt = n2; thresh = n3;
                generateLevel();
                return doLevels(currLevel + 1);
            }
            n1 = undefined; n2 = undefined; n3 = undefined;
        }
        println("Specify good weight, bad weight, threshold>=1 !");
        process.exit(1);
    }
    const rl = readline.createInterface({input: process.stdin, output: process.stdout});
    let n1;
    let n2;
    let n3;
    let responded = false;
    rl.setPrompt("good weight, bad weight, threshold: ");
    rl.prompt();

    rl.on("line", function (line) {
        if (responded) { return; }
        const chunks = line.trim().split(" ");
        if (chunks.length >= 3) {
            n1 = parseInt(chunks[0], 10);
            n2 = parseInt(chunks[1], 10);
            n3 = parseInt(chunks[2], 10);
        } else if (chunks.length === 1) {
            if (n1 === undefined) {
                n1 = parseInt(chunks[0], 10); return;
            } else if (n2 === undefined) {
                n2 = parseInt(chunks[0], 10); return;
            } else {
                n3 = parseInt(chunks[0], 10);
            }
        } else { return; }
        if (n1 >= 1 && n2 >= 1 && n3 >= 1) {
            responded = true;
            good_wt = n1;
            bad_wt = n2;
            thresh = n3;
            rl.close();
            //profiler.startProfiling("generateLevel", true);
            generateLevel();
            doLevels(currLevel + 1);
        } else {
            println("Specify good weight, bad weight, threshold>=1 !");
            n1 = undefined;
            n2 = undefined;
            n3 = undefined;
            rl.setPrompt("good weight, bad weight, threshold: ");
            rl.prompt();
        }
    });
}

function getPat(currLevel) {
    if (stdinInputLines) {
        let n1 = undefined, n2 = undefined;
        while (stdinInputLines.length > 0) {
            const chunks = stdinInputLines.shift().trim().split(" ");
            if (chunks.length >= 2) {
                n1 = parseInt(chunks[0], 10);
                n2 = parseInt(chunks[1], 10);
            } else if (chunks.length === 1) {
                if (n1 === undefined) { n1 = parseInt(chunks[0], 10); continue; }
                else { n2 = parseInt(chunks[0], 10); }
            } else { continue; }
            if (n1 >= 1 && n1 <= n2 && n2 <= max_dot) {
                pat_start = n1; pat_finish = n2;
                return getGBT(currLevel);
            }
            n1 = undefined; n2 = undefined;
        }
        println("Specify 1<=pat_start<=pat_finish<=" + max_dot + " !");
        process.exit(1);
    }
    const rl = readline.createInterface({input: process.stdin, output: process.stdout});
    let n1;
    let n2;
    let responded = false;
    rl.setPrompt("pat_start, pat_finish: ");
    rl.prompt();

    rl.on("line", function (line) {
        if (responded) { return; }
        const chunks = line.trim().split(" ");
        if (chunks.length >= 2) {
            n1 = parseInt(chunks[0], 10);
            n2 = parseInt(chunks[1], 10);
        } else if (chunks.length === 1) {
            if (n1 === undefined) {
                n1 = parseInt(chunks[0], 10); return;
            } else {
                n2 = parseInt(chunks[0], 10);
            }
        } else { return; }
        if (n1 >= 1 && n1 <= n2 && n2 <= max_dot) {
            responded = true;
            pat_start = n1;
            pat_finish = n2;
            rl.close();
            getGBT(currLevel);
        } else {
            println("Specify 1<=pat_start<=pat_finish<=" + max_dot + " !");
            n1 = undefined;
            n2 = undefined;
            rl.setPrompt("pat_start, pat_finish: ");
            rl.prompt();
        }
    });
}




function getHyph() {
    if (stdinInputLines) {
        let n1 = undefined, n2 = undefined;
        while (stdinInputLines.length > 0) {
            const chunks = stdinInputLines.shift().trim().split(" ");
            if (chunks.length >= 2) {
                n1 = parseInt(chunks[0], 10);
                n2 = parseInt(chunks[1], 10);
            } else if (chunks.length === 1) {
                if (n1 === undefined) { n1 = parseInt(chunks[0], 10); continue; }
                else { n2 = parseInt(chunks[0], 10); }
            } else { continue; }
            if (n1 >= 1 && n1 < max_val && n2 >= 1 && n2 < max_val) {
                hyph_start = n1; hyph_finish = n2;
                return doLevels(hyph_start);
            }
            n1 = undefined; n2 = undefined;
        }
        println("Specify 1<=hyph_start,hyph_finish<=" + (max_val - 1) + " !");
        process.exit(1);
    }
    const rl = readline.createInterface({input: process.stdin, output: process.stdout});
    let n1;
    let n2;
    let responded = false;
    rl.setPrompt("hyph_start, hyph_finish: ");
    rl.prompt();

    rl.on("line", function (line) {
        if (responded) { return; }
        const chunks = line.trim().split(" ");
        if (chunks.length >= 2) {
            n1 = parseInt(chunks[0], 10);
            n2 = parseInt(chunks[1], 10);
        } else if (chunks.length === 1) {
            if (n1 === undefined) {
                n1 = parseInt(chunks[0], 10); return;
            } else {
                n2 = parseInt(chunks[0], 10);
            }
        } else { return; }
        if (n1 >= 1 && n1 < max_val && n2 >= 1 && n2 < max_val) {
            responded = true;
            hyph_start = n1;
            hyph_finish = n2;
            rl.close();
            doLevels(hyph_start);
        } else {
            println("Specify 1<=hyph_start,hyph_finish<=" + (max_val - 1) + " !");
            n1 = undefined;
            n2 = undefined;
            rl.setPrompt("hyph_start, hyph_finish: ");
            rl.prompt();
        }
    });
}


/**
 * Collect characters used in dictionary and extend xext, xint and xclass
 */
function collectAndSetChars() {
    const charsS = new Set();
    let charsA;
    let cc;
    dictionary.reset();
    println("collecting chars…");
    while (!dictionary.eof()) {
        cc = dictionary.data[1].charCodeAt(dictionary.data[0]);
        if (cc === 45 || cc >= 65) {
            charsS.add(cc);
        }
        dictionary.data[0] += 1;
    }
    charsA = [];
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
    xdig = xext.slice(0, 10);
    cmax = xext.length - 1;
    cnum = xext.length;
}

function main() {
    collectAndSetChars();
    init_pattern_trie();
    read_patterns();
    procesp = true;
    hyphp = false;
    getHyph();
}

function getLeftRightHyphenMin() {
    if (stdinInputLines) {
        let n1 = undefined, n2 = undefined;
        while (stdinInputLines.length > 0) {
            const chunks = stdinInputLines.shift().trim().split(" ");
            if (chunks.length >= 2) {
                n1 = parseInt(chunks[0], 10);
                n2 = parseInt(chunks[1], 10);
            } else if (chunks.length === 1) {
                if (n1 === undefined) { n1 = parseInt(chunks[0], 10); continue; }
                else { n2 = parseInt(chunks[0], 10); }
            } else { continue; }
            if (n1 >= 1 && n1 < 15 && n2 >= 1 && n2 < 15) {
                left_hyphen_min = n1; right_hyphen_min = n2;
                return main();
            }
            n1 = undefined; n2 = undefined;
        }
        println("Specify 1<=left_hyphen_min,right_hyphen_min<=15 !");
        process.exit(1);
    }
    const rl = readline.createInterface({input: process.stdin, output: process.stdout});
    let n1;
    let n2;
    let responded = false;
    rl.setPrompt("left_hyphen_min, right_hyphen_min: ");
    rl.prompt();

    rl.on("line", function (line) {
        if (responded) { return; }
        const chunks = line.trim().split(" ");
        if (chunks.length >= 2) {
            n1 = parseInt(chunks[0], 10);
            n2 = parseInt(chunks[1], 10);
        } else if (chunks.length === 1) {
            if (n1 === undefined) {
                n1 = parseInt(chunks[0], 10); return;
            } else {
                n2 = parseInt(chunks[0], 10);
            }
        } else { return; }
        if (n1 >= 1 && n1 < 15 && n2 >= 1 && n2 < 15) {
            responded = true;
            left_hyphen_min = n1;
            right_hyphen_min = n2;
            rl.close();
            main();
        } else {
            println("Specify 1<=left_hyphen_min,right_hyphen_min<=15 !");
            n1 = undefined;
            n2 = undefined;
            rl.setPrompt("left_hyphen_min, right_hyphen_min: ");
            rl.prompt();
        }
    });
}

function init() {
    println(banner);
    initialize();
    if (!process.stdin.isTTY) {
        const rl = readline.createInterface({input: process.stdin});
        stdinInputLines = [];
        rl.on("line", function (line) {
            stdinInputLines.push(line);
        });
        rl.on("close", function () {
            getLeftRightHyphenMin();
        });
    } else {
        getLeftRightHyphenMin();
    }
}

//When filereading promises are fullfilled, start computations
Promise.all([dictionaryProm, patternInProm]).then(
    function (values) {
        dictionary = makeFile(values[0]);
        patterns = makeFile(values[1]);
        init();
    }
).catch(
    function (values) {
        println(values);
    }
);

