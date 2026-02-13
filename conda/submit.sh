#!/bin/bash
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "$0")/r-greybox" && pwd)"
STAGING_DIR="${HOME}/conda-forge-staging"

echo "=== Step 1: Fork and clone staged-recipes ==="
if [ -d "${STAGING_DIR}" ]; then
    echo "Directory ${STAGING_DIR} already exists, pulling latest..."
    cd "${STAGING_DIR}"
    git checkout main
    git pull upstream main 2>/dev/null || git pull origin main
else
    gh repo fork conda-forge/staged-recipes --clone --clone-dir "${STAGING_DIR}"
    cd "${STAGING_DIR}"
    git remote add upstream https://github.com/conda-forge/staged-recipes.git 2>/dev/null || true
fi

echo "=== Step 2: Create branch and copy recipe ==="
git checkout -b r-greybox main 2>/dev/null || git checkout r-greybox
cp -r "${RECIPE_DIR}" recipes/r-greybox

echo "=== Step 3: Commit and push ==="
git add recipes/r-greybox
git commit -m "Add r-greybox recipe (v2.0.7 from CRAN)"
git push -u origin r-greybox

echo "=== Step 4: Create PR ==="
gh pr create \
    --repo conda-forge/staged-recipes \
    --title "Add r-greybox" \
    --body "$(cat <<'EOF'
## Summary
- CRAN package `greybox` v2.0.7
- Toolbox for regression model building and forecasting
- License: LGPL-2.1
- Required as a dependency for `r-smooth`

## Checklist
- [x] Source from CRAN with archive fallback
- [x] All dependencies available on conda-forge
- [x] Build scripts for Linux/macOS and Windows
- [x] Test: `library('greybox')`
EOF
)"

echo "=== Done! PR created. Wait for merge before submitting r-smooth. ==="
