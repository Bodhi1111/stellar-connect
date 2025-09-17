#!/bin/bash

# BMad Method Update Checker
echo "🔍 Checking BMad Method version status..."
echo ""

# Get current version
CURRENT=$(npx bmad-method --version 2>/dev/null || echo "Not found")
echo "📦 Current Version: $CURRENT"

# Get latest version
LATEST=$(npm view bmad-method version 2>/dev/null || echo "Unable to check")
echo "🚀 Latest Version:  $LATEST"
echo ""

# Compare versions
if [ "$CURRENT" = "$LATEST" ]; then
    echo "✅ You're up to date!"
elif [ "$CURRENT" = "Not found" ]; then
    echo "❌ BMad Method not found. Run: npx bmad-method install"
elif [ "$LATEST" = "Unable to check" ]; then
    echo "⚠️  Unable to check for updates. Check your internet connection."
else
    echo "🔄 Update available!"
    echo ""
    echo "To upgrade, run one of:"
    echo "  npx bmad-method@latest install --upgrade"
    echo "  npx bmad-method@latest install --full --directory . --ide cursor"
fi

echo ""
echo "💡 For detailed upgrade instructions, see: UPGRADE_GUIDE.md"
