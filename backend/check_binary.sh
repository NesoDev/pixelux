#!/bin/bash
# Binary verification script for Pixelux API container
# Verifies that the pixelart_mpi binary exists and has all required dependencies

set -e

BINARY="${PIXELART_BINARY:-/home/mpiuser/shared/pixelart/pixelart_mpi}"

echo "=== Pixelux Binary Verification ==="
echo "Checking binary: $BINARY"

# Check if binary exists
if [ ! -f "$BINARY" ]; then
    echo "❌ ERROR: Binary not found at $BINARY"
    echo "Please ensure the shared volume is mounted correctly and the binary has been compiled."
    exit 1
fi

# Check if binary is executable
if [ ! -x "$BINARY" ]; then
    echo "⚠️  WARNING: Binary exists but is not executable, attempting to fix permissions..."
    chmod +x "$BINARY" 2>/dev/null || {
        echo "❌ ERROR: Cannot make binary executable. Permission denied."
        exit 1
    }
fi

# Check library dependencies
echo ""
echo "Checking library dependencies..."
if command -v ldd &> /dev/null; then
    MISSING_LIBS=$(ldd "$BINARY" 2>&1 | grep "not found" || true)
    if [ -n "$MISSING_LIBS" ]; then
        echo "❌ ERROR: Missing required libraries:"
        echo "$MISSING_LIBS"
        echo ""
        echo "Full library list:"
        ldd "$BINARY"
        exit 1
    else
        echo "✅ All required libraries are present"
        echo ""
        echo "Library details:"
        ldd "$BINARY" | head -n 15
        if [ $(ldd "$BINARY" | wc -l) -gt 15 ]; then
            echo "... ($(ldd "$BINARY" | wc -l) total libraries)"
        fi
    fi
else
    echo "⚠️  WARNING: ldd command not available, skipping library check"
fi

# Test binary execution (quick help check)
echo ""
echo "Testing binary execution..."
if timeout 5 "$BINARY" --help &>/dev/null || timeout 5 "$BINARY" -h &>/dev/null || [ $? -eq 1 ]; then
    echo "✅ Binary is executable (may not have --help flag, but responds)"
else
    echo "⚠️  WARNING: Binary responds with error, but may still work for image processing"
fi

echo ""
echo "=== Binary Verification Complete ==="
echo "Binary: $BINARY"
echo "Status: Ready"
echo ""
