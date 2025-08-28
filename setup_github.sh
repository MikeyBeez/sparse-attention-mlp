#!/bin/bash

# Create GitHub repository for sparse attention project
# This script helps set up the remote repository

echo "🚀 Setting up GitHub repository for Sparse Attention MLP"
echo "=================================================="

# Check if gh CLI is available
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) not found. Please install it first:"
    echo "   brew install gh"
    echo "   gh auth login"
    exit 1
fi

# Create the repository on GitHub
echo "📝 Creating repository on GitHub..."
gh repo create sparse-attention-mlp \
    --public \
    --description "Sparse attention with MLP approximation - demonstrating computational benefits at scale" \
    --add-readme=false

if [ $? -eq 0 ]; then
    echo "✅ Repository created successfully!"
    
    # Add remote and push
    echo "🔗 Adding remote and pushing..."
    git remote add origin https://github.com/$(gh api user --jq .login)/sparse-attention-mlp.git
    git branch -M main
    git push -u origin main
    
    if [ $? -eq 0 ]; then
        echo "✅ Repository pushed successfully!"
        echo "🌐 Repository URL: https://github.com/$(gh api user --jq .login)/sparse-attention-mlp"
        echo ""
        echo "🎯 Next steps:"
        echo "   1. Visit the repository URL above"
        echo "   2. Add any additional collaborators"
        echo "   3. Consider adding topics/tags for discoverability"
        echo "   4. Share your research findings!"
    else
        echo "❌ Failed to push to remote repository"
    fi
else
    echo "❌ Failed to create repository"
    echo "💡 Alternative: Create repository manually at https://github.com/new"
    echo "   Then run: git remote add origin https://github.com/yourusername/sparse-attention-mlp.git"
    echo "           git branch -M main"  
    echo "           git push -u origin main"
fi
