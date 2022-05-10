# VSCode Rat Extension

This is the vscode extension for the [rat](https://github.com/bbbales2/regressions)
programming language.

## Build and Install

I don't know how to do this yet

## Development

To develop on this extension:
1. Make sure you're using a terminal where the `rat-language-server` is defined.

  The client extension needs to be able to launch the rat language server (which is
  a command line utility in the rat package itself). For debugging, the easiest way
  to get this to work is, if you launch vscode in an environment where `PATH` contains
  the language server, then we can configure vscode (see `.vscode/launch.json`) to
  pass this `PATH` to the vscode client that gets launched for debugging (this is
  a separate client from the first).

  The easiest way to get the `rat-language-server` command line utility installed but
  also be able to make changes on it and see how they're reflected (it is convenient
  to developt the client and server together) is to install the rat package with
  `pip install -e .` (the `e` makes it so that you can modify the python package in-place
  and these changes are reflected in whatever Python environment you're using -- this
  includes command line utilities installed with Python packages).

2. Navigate into the vscode-rat-extension source folder `cd vs-code-rat-extension`

3. Build and install dependencies with `npm install`

  As a fair warning, I'm not actually sure how npm/typescript/javascript all work together.
  I largely copy-pasted all this from Microsoft examples
  [here](https://code.visualstudio.com/api/language-extensions/language-server-extension-guide).

4. Launch a session of vscode in this folder with `code .`

  In the end if you're developing the server and client you'll have 3 vscode sessions, one
  to edit the server, one to edit the client, and one running the client/server combo so
  you can see what is happening.

  Right now the `rat-language-server` writes a log, `rat-language-server.json` to wherever it
  is running. The default `Launch Client` debug configuration will launch the client in the
  `vscode-rat-extension/test_workspace` folder, so when the client is running you can look there
  for logs. It is possible to attach a vscode Python debugger to the language server as it is
  running by using the `Attach to Process` debug configuration and selecting the language server
  process.

## Credits

This is modified from the lsp-example code
[here](https://code.visualstudio.com/api/language-extensions/language-server-extension-guide).

