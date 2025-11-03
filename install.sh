#!/bin/bash
# Jarvis Stock Scanner Installation Script

echo "🚀 Installing Jarvis Stock Scanner..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install pandas numpy yfinance

# Make scripts executable
chmod +x jarvis

# Copy to local bin
mkdir -p ~/.local/bin
cp jarvis ~/.local/bin/

# Add to PATH if not already there
if ! echo $PATH | grep -q "$HOME/.local/bin"; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    echo "✅ Added ~/.local/bin to PATH in ~/.bashrc"
fi

# Create alias
if ! grep -q "alias jarvis-scan" ~/.bashrc; then
    echo 'alias jarvis-scan="jarvis"' >> ~/.bashrc
    echo "✅ Added jarvis-scan alias"
fi

echo "✅ Installation complete!"
echo "📝 Run 'source ~/.bashrc' or open a new terminal"
echo "🎯 Then use: jarvis, jarvis focus, jarvis analyze AAPL, etc."
