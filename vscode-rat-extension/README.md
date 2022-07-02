To get syntax hightlighting in vscode you'll need to build and install this
package manually.

The steps are described in the
[Visual Studio docs](https://code.visualstudio.com/api/working-with-extensions/publishing-extension).

```
# Navigate into the directory with the `package.json`
# Install Visual Studio Code Extensions package
npm install vsce
# Build this package
./node_modules/vsce/vsce package
# Install the package
code --install-extension rat-language-client-0.0.1.vsix
```

That should be it. You might need restart vscode to get everything running.