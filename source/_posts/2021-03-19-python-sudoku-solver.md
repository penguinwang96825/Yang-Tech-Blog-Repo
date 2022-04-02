---
title: Python Sudoku Solver
top: false
cover: false
toc: true
mathjax: true
date: 2021-03-19 23:09:29
img: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2021/03/19/2021-03-19-python-sudoku-solver/wallhaven-6krvpw.jpg?raw=true
coverImg: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2021/03/19/2021-03-19-python-sudoku-solver/wallhaven-6krvpw.jpg?raw=true
summary: Sudoku is a logic-based, combinatorial number-placement puzzle. In classic sudoku, the objective is to fill a 9×9 grid with digits so that each column, each row, and each of the nine 3×3 subgrids that compose the grid (also called "boxes", "blocks", or "regions") contain all of the digits from 1 to 9. The puzzle setter provides a partially completed grid, which for a well-posed puzzle has a single solution.
tags:
    - Python
    - Javascript
    - Algorithm
categories: Mathematics
---

# Introduction

Sudoku is a logic-based, combinatorial number-placement puzzle. In classic sudoku, the objective is to fill a 9×9 grid with digits so that each column, each row, and each of the nine 3×3 subgrids that compose the grid (also called "boxes", "blocks", or "regions") contain all of the digits from 1 to 9. The puzzle setter provides a partially completed grid, which for a well-posed puzzle has a single solution.

# Mathematics of Sudoku

The general problem of solving Sudoku puzzles on {% mathjax %} n^2 \times n^2 {% endmathjax %} grids of {% mathjax %} n \times n {% endmathjax %} blocks is known to be [NP-complete](https://en.wikipedia.org/wiki/NP-completeness). Many computer algorithms, such as backtracking and dancing links can solve most 9×9 puzzles efficiently, but combinatorial explosion occurs as n increases, creating limits to the properties of Sudokus that can be constructed, analyzed, and solved as n increases. A Sudoku puzzle can be expressed as a [graph coloring](https://en.wikipedia.org/wiki/Graph_coloring) problem. The aim is to construct a 9-coloring of a particular graph, given a partial 9-coloring. The number of classic 9×9 Sudoku solution grids is 6,670,903,752,021,072,936,960 (sequence [A107739](https://oeis.org/A107739) in the OEIS), or around {% mathjax %} 6.67 \times 10^{21} {% endmathjax %}. This is roughly {% mathjax %} 1.2 \times 10^{-6} {% endmathjax %} times the number of 9×9 Latin squares.

# Python Implementation

The code below is referred from [here](https://www.youtube.com/watch?v=G_UYXzGuqvM&list=PLCZeVeoafktVGu9rvM9PHrAdrsUURtLTo&index=59&t=454s&ab_channel=Computerphile).

```python
grid = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0], 
    [6, 0, 0, 1, 9, 5, 0, 0, 0], 
    [0, 9, 8, 0, 0, 0, 0, 6, 0], 
    [8, 0, 0, 0, 6, 0, 0, 0, 3], 
    [4, 0, 0, 8, 0, 3, 0, 0, 1], 
    [7, 0, 0, 0, 2, 0, 0, 0, 6], 
    [0, 6, 0, 0, 0, 0, 2, 8, 0], 
    [0, 0, 0, 4, 1, 9, 0, 0, 5], 
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

def possible(x, y, n, grid):
    for i in range(9):
        if grid[i][x] == n:
            return False
    for i in range(9):
        if grid[y][i] == n:
            return False
    x0 = (x//3) * 3
    y0 = (y//3) * 3
    for i in range(3):
        for j in range(3):
            if grid[y0+i][x0+j] == n:
                return False
    return True

def solve(grid):
    for x in range(9):
        for y in range(9):
            if grid[y][x] == 0:
                for n in range(1, 10):
                    if possible(x, y, n, grid):
                        grid[y][x] = n
                        solve(grid)
                        grid[y][x] = 0
                return
    print(np.matrix(grid))
    input("More?")

solve(grid)
```

# Sudoku Solver

Enter the numbers of the puzzle you want to solve in the grid.

<div class="playground" style="display: flex;justify-content: center;">
    <div id="container"></div>
</div>
<div id="menu" class="sudoku-menu">
    <button class="sudoku-b" id="solve" style="background-color:#ccc">Solve</button>
    <button class="sudoku-b" id="validate" style="background-color:#ccc">Validate Board</button>
    <button class="sudoku-b" id="reset" style="background-color:#ccc">Reset</button>
</div>

<script type="text/javascript">
    /**
     * A Javascript implementation of a Sudoku game, including a
     * backtracking algorithm solver. For example usage see the
     * attached index.html demo.
     *
     * @author Moriel Schottlender
     */
    var Sudoku = ( function ( $ ){
        var _instance, _game,
            /**
             * Default configuration options. These can be overriden
             * when loading a game instance.
             * @property {Object}
             */
            defaultConfig = {
                // If set to true, the game will validate the numbers
                // as the player inserts them. If it is set to false,
                // validation will only happen at the end.
                'validate_on_insert': true,
                // If set to true, the system will display the elapsed
                // time it took for the solver to finish its operation.
                'show_solver_timer': true,
                // If set to true, the recursive solver will count the
                // number of recursions and backtracks it performed and
                // display them in the console.
                'show_recursion_counter': true,
                // If set to true, the solver will test a shuffled array
                // of possible numbers in each empty input box.
                // Otherwise, the possible numbers are ordered, which
                // means the solver will likely give the same result
                // when operating in the same game conditions.
                'solver_shuffle_numbers': true
            },
            paused = false,
            counter = 0;

        /**
         * Initialize the singleton
         * @param {Object} config Configuration options
         * @returns {Object} Singleton methods
         */
        function init( config ) {
            conf = $.extend( {}, defaultConfig, config );
            _game = new Game( conf );

            /** Public methods **/
            return {
                /**
                 * Return a visual representation of the board
                 * @returns {jQuery} Game table
                 */
                getGameBoard: function() {
                    return _game.buildGUI();
                },

                /**
                 * Reset the game board.
                 */
                reset: function() {
                    _game.resetGame();
                },

                /**
                 * Call for a validation of the game board.
                 * @returns {Boolean} Whether the board is valid
                 */
                validate: function() {
                    var isValid;

                    isValid = _game.validateMatrix();
                    $( '.sudoku-container' ).toggleClass( 'valid-matrix', isValid );
                },

                /**
                 * Call for the solver routine to solve the current
                 * board.
                 */
                solve: function() {
                    var isValid, starttime, endtime, elapsed;
                    // Make sure the board is valid first
                    if ( !_game.validateMatrix() ) {
                        return false;
                    }
                    // Reset counters
                    _game.recursionCounter = 0;
                    _game.backtrackCounter = 0;

                    // Check start time
                    starttime = Date.now();

                    // Solve the game
                    isValid = _game.solveGame( 0, 0 );

                    // Get solving end time
                    endtime = Date.now();

                    // Visual indication of whether the game was solved
                    $( '.sudoku-container' ).toggleClass( 'valid-matrix', isValid );
                    if ( isValid ) {
                        $( '.valid-matrix input' ).attr( 'disabled', 'disabled' );
                    }

                    // Display elapsed time
                    if ( _game.config.show_solver_timer ) {
                        elapsed = endtime - starttime;
                        window.console.log( 'Solver elapsed time: ' + elapsed + 'ms' );
                    }
                    // Display number of reursions and backtracks
                    if ( _game.config.show_recursion_counter ) {
                        window.console.log( 'Solver recursions: ' + _game.recursionCounter );
                        window.console.log( 'Solver backtracks: ' + _game.backtrackCounter );
                    }
                }
            };
        }

        /**
         * Sudoku singleton engine
         * @param {Object} config Configuration options
         */
        function Game( config ) {
            this.config = config;

            // Initialize game parameters
            this.recursionCounter = 0;
            this.$cellMatrix = {};
            this.matrix = {};
            this.validation = {};

            this.resetValidationMatrices();
            return this;
        }
        /**
         * Game engine prototype methods
         * @property {Object}
         */
        Game.prototype = {
            /**
             * Build the game GUI
             * @returns {jQuery} Table containing 9x9 input matrix
             */
            buildGUI: function() {
                var $td, $tr,
                    $table = $( '<table>' )
                        .addClass( 'sudoku-container' );

                for ( var i = 0; i < 9; i++ ) {
                    $tr = $( '<tr>' );
                    this.$cellMatrix[i] = {};

                    for ( var j = 0; j < 9; j++ ) {
                        // Build the input
                        this.$cellMatrix[i][j] = $( '<input>' )
                            .attr( 'maxlength', 1 )
                            .data( 'row', i )
                            .data( 'col', j )
                            .on( 'keyup', $.proxy( this.onKeyUp, this ) );

                        $td = $( '<td>' ).append( this.$cellMatrix[i][j] );
                        // Calculate section ID
                        sectIDi = Math.floor( i / 3 );
                        sectIDj = Math.floor( j / 3 );
                        // Set the design for different sections
                        if ( ( sectIDi + sectIDj ) % 2 === 0 ) {
                            $td.addClass( 'sudoku-section-one' );
                        } else {
                            $td.addClass( 'sudoku-section-two' );
                        }
                        // Build the row
                        $tr.append( $td );
                    }
                    // Append to table
                    $table.append( $tr );
                }
                // Return the GUI table
                return $table;
            },

            /**
             * Handle keyup events.
             *
             * @param {jQuery.event} e Keyup event
             */
            onKeyUp: function( e ) {
                var sectRow, sectCol, secIndex,
                    starttime, endtime, elapsed,
                    isValid = true,
                    val = $.trim( $( e.currentTarget ).val() ),
                    row = $( e.currentTarget ).data( 'row' ),
                    col = $( e.currentTarget ).data( 'col' );

                // Reset board validation class
                $( '.sudoku-container' ).removeClass( 'valid-matrix' );

                // Validate, but only if validate_on_insert is set to true
                if ( this.config.validate_on_insert ) {
                    isValid = this.validateNumber( val, row, col, this.matrix.row[row][col] );
                    // Indicate error
                    $( e.currentTarget ).toggleClass( 'sudoku-input-error', !isValid );
                }

                // Calculate section identifiers
                sectRow = Math.floor( row / 3 );
                sectCol = Math.floor( col / 3 );
                secIndex = ( row % 3 ) * 3 + ( col % 3 );

                // Cache value in matrix
                this.matrix.row[row][col] = val;
                this.matrix.col[col][row] = val;
                this.matrix.sect[sectRow][sectCol][secIndex] = val;
            },

            /**
             * Reset the board and the game parameters
             */
            resetGame: function() {
                this.resetValidationMatrices();
                for ( var row = 0; row < 9; row++ ) {
                    for ( var col = 0; col < 9; col++ ) {
                        // Reset GUI inputs
                        this.$cellMatrix[row][col].val( '' );
                    }
                }

                $( '.sudoku-container input' ).removeAttr( 'disabled' );
                $( '.sudoku-container' ).removeClass( 'valid-matrix' );
            },

            /**
             * Reset and rebuild the validation matrices
             */
            resetValidationMatrices: function() {
                this.matrix = { 'row': {}, 'col': {}, 'sect': {} };
                this.validation = { 'row': {}, 'col': {}, 'sect': {} };

                // Build the row/col matrix and validation arrays
                for ( var i = 0; i < 9; i++ ) {
                    this.matrix.row[i] = [ '', '', '', '', '', '', '', '', '' ];
                    this.matrix.col[i] = [ '', '', '', '', '', '', '', '', '' ];
                    this.validation.row[i] = [];
                    this.validation.col[i] = [];
                }

                // Build the section matrix and validation arrays
                for ( var row = 0; row < 3; row++ ) {
                    this.matrix.sect[row] = [];
                    this.validation.sect[row] = {};
                    for ( var col = 0; col < 3; col++ ) {
                        this.matrix.sect[row][col] = [ '', '', '', '', '', '', '', '', '' ];
                        this.validation.sect[row][col] = [];
                    }
                }
            },

            /**
             * Validate the current number that was inserted.
             *
             * @param {String} num The value that is inserted
             * @param {Number} rowID The row the number belongs to
             * @param {Number} colID The column the number belongs to
             * @param {String} oldNum The previous value
             * @returns {Boolean} Valid or invalid input
             */
            validateNumber: function( num, rowID, colID, oldNum ) {
                var isValid = true,
                    // Section
                    sectRow = Math.floor( rowID / 3 ),
                    sectCol = Math.floor( colID / 3 );

                // This is given as the matrix component (old value in
                // case of change to the input) in the case of on-insert
                // validation. However, in the solver, validating the
                // old number is unnecessary.
                oldNum = oldNum || '';

                // Remove oldNum from the validation matrices,
                // if it exists in them.
                if ( this.validation.row[rowID].indexOf( oldNum ) > -1 ) {
                    this.validation.row[rowID].splice(
                        this.validation.row[rowID].indexOf( oldNum ), 1
                    );
                }
                if ( this.validation.col[colID].indexOf( oldNum ) > -1 ) {
                    this.validation.col[colID].splice(
                        this.validation.col[colID].indexOf( oldNum ), 1
                    );
                }
                if ( this.validation.sect[sectRow][sectCol].indexOf( oldNum ) > -1 ) {
                    this.validation.sect[sectRow][sectCol].splice(
                        this.validation.sect[sectRow][sectCol].indexOf( oldNum ), 1
                    );
                }
                // Skip if empty value

                if ( num !== '' ) {


                    // Validate value
                    if (
                        // Make sure value is numeric
                        $.isNumeric( num ) &&
                        // Make sure value is within range
                        Number( num ) > 0 &&
                        Number( num ) <= 9
                    ) {
                        // Check if it already exists in validation array
                        if (
                            $.inArray( num, this.validation.row[rowID] ) > -1 ||
                            $.inArray( num, this.validation.col[colID] ) > -1 ||
                            $.inArray( num, this.validation.sect[sectRow][sectCol] ) > -1
                        ) {
                            isValid = false;
                        } else {
                            isValid = true;
                        }
                    }

                    // Insert new value into validation array even if it isn't
                    // valid. This is on purpose: If there are two numbers in the
                    // same row/col/section and one is replaced, the other still
                    // exists and should be reflected in the validation.
                    // The validation will keep records of duplicates so it can
                    // remove them safely when validating later changes.
                    this.validation.row[rowID].push( num );
                    this.validation.col[colID].push( num );
                    this.validation.sect[sectRow][sectCol].push( num );
                }

                return isValid;
            },

            /**
             * Validate the entire matrix
             * @returns {Boolean} Valid or invalid matrix
             */
            validateMatrix: function() {
                var isValid, val, $element,
                    hasError = false;

                // Go over entire board, and compare to the cached
                // validation arrays
                for ( var row = 0; row < 9; row++ ) {
                    for ( var col = 0; col < 9; col++ ) {
                        val = this.matrix.row[row][col];
                        // Validate the value
                        isValid = this.validateNumber( val, row, col, val );
                        this.$cellMatrix[row][col].toggleClass( 'sudoku-input-error', !isValid );
                        if ( !isValid ) {
                            hasError = true;
                        }
                    }
                }
                return !hasError;
            },

            /**
             * A recursive 'backtrack' solver for the
             * game. Algorithm is based on the StackOverflow answer
             * http://stackoverflow.com/questions/18168503/recursively-solving-a-sudoku-puzzle-using-backtracking-theoretically
             */
            solveGame: function( row, col ) {
                var cval, sqRow, sqCol, $nextSquare, legalValues,
                    sectRow, sectCol, secIndex, gameResult;

                this.recursionCounter++;
                $nextSquare = this.findClosestEmptySquare( row, col );
                if ( !$nextSquare ) {
                    // End of board
                    return true;
                } else {
                    sqRow = $nextSquare.data( 'row' );
                    sqCol = $nextSquare.data( 'col' );
                    legalValues = this.findLegalValuesForSquare( sqRow, sqCol );

                    // Find the segment id
                    sectRow = Math.floor( sqRow / 3 );
                    sectCol = Math.floor( sqCol / 3 );
                    secIndex = ( sqRow % 3 ) * 3 + ( sqCol % 3 );

                    // Try out legal values for this cell
                    for ( var i = 0; i < legalValues.length; i++ ) {
                        cval = legalValues[i];
                        // Update value in input
                        $nextSquare.val( cval );
                        // Update in matrices
                        this.matrix.row[sqRow][sqCol] = cval;
                        this.matrix.col[sqCol][sqRow] = cval;
                        this.matrix.sect[sectRow][sectCol][secIndex] = cval;

                        // Recursively keep trying
                        if ( this.solveGame( sqRow, sqCol ) ) {
                            return true;
                        } else {
                            // There was a problem, we should backtrack
                            this.backtrackCounter++;

                            // Remove value from input
                            this.$cellMatrix[sqRow][sqCol].val( '' );
                            // Remove value from matrices
                            this.matrix.row[sqRow][sqCol] = '';
                            this.matrix.col[sqCol][sqRow] = '';
                            this.matrix.sect[sectRow][sectCol][secIndex] = '';
                        }
                    }
                    // If there was no success with any of the legal
                    // numbers, call backtrack recursively backwards
                    return false;
                }
            },

            /**
             * Find closest empty square relative to the given cell.
             *
             * @param {Number} row Row id
             * @param {Number} col Column id
             * @returns {jQuery} Input element of the closest empty
             *  square
             */
            findClosestEmptySquare: function( row, col ) {
                var walkingRow, walkingCol, found = false;
                for ( var i = ( col + 9*row ); i < 81; i++ ) {
                    walkingRow = Math.floor( i / 9 );
                    walkingCol = i % 9;
                    if ( this.matrix.row[walkingRow][walkingCol] === '' ) {
                        found = true;
                        return this.$cellMatrix[walkingRow][walkingCol];
                    }
                }
            },

            /**
             * Find the available legal numbers for the square in the
             * given row and column.
             *
             * @param {Number} row Row id
             * @param {Number} col Column id
             * @returns {Array} An array of available numbers
             */
            findLegalValuesForSquare: function( row, col ) {
                var legalVals, legalNums, val, i,
                    sectRow = Math.floor( row / 3 ),
                    sectCol = Math.floor( col / 3 );

                legalNums = [ 1, 2, 3, 4, 5, 6, 7, 8, 9];

                // Check existing numbers in col
                for ( i = 0; i < 9; i++ ) {
                    val = Number( this.matrix.col[col][i] );
                    if ( val > 0 ) {
                        // Remove from array
                        if ( legalNums.indexOf( val ) > -1 ) {
                            legalNums.splice( legalNums.indexOf( val ), 1 );
                        }
                    }
                }

                // Check existing numbers in row
                for ( i = 0; i < 9; i++ ) {
                    val = Number( this.matrix.row[row][i] );
                    if ( val > 0 ) {
                        // Remove from array
                        if ( legalNums.indexOf( val ) > -1 ) {
                            legalNums.splice( legalNums.indexOf( val ), 1 );
                        }
                    }
                }

                // Check existing numbers in section
                sectRow = Math.floor( row / 3 );
                sectCol = Math.floor( col / 3 );
                for ( i = 0; i < 9; i++ ) {
                    val = Number( this.matrix.sect[sectRow][sectCol][i] );
                    if ( val > 0 ) {
                        // Remove from array
                        if ( legalNums.indexOf( val ) > -1 ) {
                            legalNums.splice( legalNums.indexOf( val ), 1 );
                        }
                    }
                }

                if ( this.config.solver_shuffle_numbers ) {
                    // Shuffling the resulting 'legalNums' array will
                    // make sure the solver produces different answers
                    // for the same scenario. Otherwise, 'legalNums'
                    // will be chosen in sequence.
                    for ( i = legalNums.length - 1; i > 0; i-- ) {
                        var rand = getRandomInt( 0, i );
                        temp = legalNums[i];
                        legalNums[i] = legalNums[rand];
                        legalNums[rand] = temp;
                    }
                }

                return legalNums;
            },
        };

        /**
         * Get a random integer within a range
         *
         * @param {Number} min Minimum number
         * @param {Number} max Maximum range
         * @returns {Number} Random number within the range (Inclusive)
         */
        function getRandomInt(min, max) {
            return Math.floor( Math.random() * ( max + 1 ) ) + min;
        }

        return {
            /**
             * Get the singleton instance. Only one instance is allowed.
             * The method will either create an instance or will return
             * the already existing instance.
             *
             * @param {[type]} config [description]
             * @returns {[type]} [description]
             */
            getInstance: function( config ) {
                if ( !_instance ) {
                    _instance = init( config );
                }
                return _instance;
            }
        };
    } )( jQuery );


    $( document ).ready( function() {
                var game = Sudoku.getInstance();
                $( '#container').append( game.getGameBoard() );
                $( '#solve').click( function() {
                    game.solve();
                } );
                $( '#validate').click( function() {
                    game.validate();
                } );
                $( '#reset').click( function() {
                    game.reset();
                } );
            } );
</script>

# Conclusion

Sudoku is a 'brain game' that requires a variety of cognitive skills, such as quick decision making, spotting patterns, and applying logical reasoning. Sudoku is a nice escape from the little challenges of daily life – we can solve a Sudoku puzzle and point to it as a tangible example of what we can achieve with our minds.

## Reference

1. https://www.youtube.com/watch?v=G_UYXzGuqvM&list=PLCZeVeoafktVGu9rvM9PHrAdrsUURtLTo&index=59&t=454s&ab_channel=Computerphile
2. https://en.wikipedia.org/wiki/Sudoku
3. https://sudoku.com/
4. https://codereview.stackexchange.com/questions/239248/sudoku-game-in-javascript
5. https://sudokuspoiler.azurewebsites.net/
6. https://www.sudoku-solutions.com/