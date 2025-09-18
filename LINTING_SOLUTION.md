# Linting Configuration - Stellar Connect

## Why Were We Getting Excessive Linting Errors?

The markdown linting errors were caused by:

1. **BMad Method QA Focus**: The BMad framework includes quality assurance agents and configurations that enable strict linting rules
2. **Cursor IDE Integration**: The BMad installation configured Cursor with enhanced linting rules
3. **New Project Setup**: Unlike your previous projects, this one had strict markdown formatting rules enabled

## Solution Applied

### Files Created:
- `.markdownlint.json` - JSON configuration to disable overly strict rules
- `.markdownlint.yaml` - YAML configuration with detailed comments
- `.vscode/settings.json` - Cursor/VSCode specific settings

### Rules Disabled:
- `MD022` - Blanks around headings (too strict for documentation)
- `MD032` - Blanks around lists (interferes with natural writing)
- `MD031` - Blanks around code fences (unnecessary spacing)
- `MD026` - Trailing punctuation in headings (colons are useful)
- `MD034` - Bare URLs (sometimes needed for documentation)
- `MD040` - Code block language specification (not always needed)
- `MD009` - Trailing spaces (minor formatting issue)
- `MD012` - Multiple blank lines (natural spacing)
- `MD047` - Single trailing newline (file ending preference)

### Rules Still Enabled:
- Heading hierarchy and consistency
- Proper list formatting and indentation
- Link syntax correctness
- Code block consistency
- Emphasis and strong text formatting
- Image alt text requirements
- No broken links or syntax errors

## Result

✅ **No more excessive linting errors** while maintaining code quality
✅ **Natural markdown writing** without forced formatting
✅ **Important quality checks** still active
✅ **Compatible with your normal workflow** from other projects

## For Future Projects

You can copy these configuration files to any new project to avoid similar issues:
```bash
cp .markdownlint.json /path/to/new/project/
cp .vscode/settings.json /path/to/new/project/.vscode/
```

---
*Configured: September 17, 2025*
*Status: ✅ Linting normalized to reasonable standards*
