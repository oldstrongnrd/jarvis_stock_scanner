#!/bin/bash
# Big 3 Scanner Installation Script

echo "🚀 Installing Big 3 Scanner..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --user -r requirements_enhanced.txt

# Make scripts executable
chmod +x big3

# Copy to local bin
mkdir -p ~/.local/bin
cp big3 ~/.local/bin/

# Add to PATH if not already there
if ! echo $PATH | grep -q "$HOME/.local/bin"; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    echo "✅ Added ~/.local/bin to PATH in ~/.bashrc"
fi

# Create alias
if ! grep -q "alias big3scanner" ~/.bashrc; then
    echo 'alias big3scanner="big3"' >> ~/.bashrc
    echo "✅ Added big3scanner alias"
fi

echo "✅ Installation complete!"
echo "📝 Run 'source ~/.bashrc' or open a new terminal"
echo "🎯 Then use: big3, big3 focus, big3 analyze AAPL, etc."
