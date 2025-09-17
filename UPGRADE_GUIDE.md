# BMad Method Upgrade Guide

## Current Installation
- **Version**: 4.43.1
- **Status**: ✅ Latest Version
- **Install Date**: September 17, 2025
- **Type**: Full installation with Cursor integration

## How to Check for Updates

### 1. Check Current Version
```bash
npx bmad-method --version
```

### 2. Check Latest Available Version
```bash
npm view bmad-method version
```

### 3. Compare Versions
If the npm version is higher than your current version, an upgrade is available.

## Upgrade Methods

### Method 1: Smart Upgrade (Recommended)
```bash
# This preserves your existing configuration
npx bmad-method@latest install --upgrade
```

### Method 2: Full Reinstall
```bash
# This will detect and upgrade your existing installation
npx bmad-method@latest install --full --directory . --ide cursor
```

### Method 3: Interactive Upgrade
```bash
# Run without flags for interactive prompts
npx bmad-method@latest install
```

## What Gets Upgraded

✅ **Preserved During Upgrade:**
- Your project files and custom code
- Git history and commits
- Custom configurations you've made

🔄 **Updated During Upgrade:**
- BMad agent definitions (.bmad-core/agents/)
- Framework templates (.bmad-core/templates/)
- Workflow definitions (.bmad-core/workflows/)
- Cursor IDE rules (.cursor/rules/bmad/)
- Core framework files

## Post-Upgrade Verification

1. **Check Version**:
   ```bash
   npx bmad-method --version
   ```

2. **Verify Installation**:
   ```bash
   cat .bmad-core/install-manifest.yaml | head -5
   ```

3. **Test Agent Activation**:
   - Try activating an agent like `@bmad-master`
   - Verify Cursor rules are working

## Backup Before Upgrade (Optional)

```bash
# Create a backup branch
git checkout -b backup-before-upgrade
git add .
git commit -m "Backup before BMad upgrade"
git checkout main
```

## Rollback (If Needed)

```bash
# If upgrade causes issues, rollback
git checkout backup-before-upgrade
# Or restore from specific commit
git reset --hard <commit-hash>
```

---
*Last Updated: September 17, 2025*
*Current Version: 4.43.1 (Latest)*
