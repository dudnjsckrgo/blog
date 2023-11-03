echo "Deploying updates to GitHub..."

# Build the project.
hugo -t blowfish

# Go To Public folder
cd public
# Add changes to git.
git add .

# Commit changes.
$msg="rebuilding site $(Get-Date)"
if ($args.Count -eq 1) {
    $msg=$args[0]
}
git commit -m "$msg"

# Push source and build repos.
git push origin master

# Come Back up to the Project Root
cd ..

# blog repository Commit & Push
git add .

$msg="rebuilding site $(Get-Date)"
if ($args.Count -eq 1) {
    $msg=$args[0]
}
git commit -m "$msg"

git push origin master
