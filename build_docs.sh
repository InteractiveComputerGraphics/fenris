#!/bin/bash

# This script injects a header that lets us write our Rust documentation with TeX formulas rendered by KaTeX.
# This approach only works when providing `cargo doc` with an **absolute path** to the header file, hence
# we obtain the path relative to this script file and build the docs with this header injection.

# From: https://stackoverflow.com/a/246128
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
HEADER_PATH="$SCRIPT_DIR/assets/doc-header.html"
BEFORE_CONTENT_PATH="$SCRIPT_DIR/assets/doc-before-content.html"

# Copy assets to other crates in the workspace, so that
# they can use the same assets for docs.rs
cp $HEADER_PATH "$SCRIPT_DIR/fenris-solid/assets/doc-header.html"
cp $BEFORE_CONTENT_PATH "$SCRIPT_DIR/fenris-solid/assets/doc-before-content.html"

# Note: command line arguments are forwarded to the final invocation
RUSTDOCFLAGS="--html-in-header $HEADER_PATH --html-before-content $BEFORE_CONTENT_PATH" cargo doc $@
