# Aplx Dock v1.6

<video src="https://github.com/user-attachments/assets/94dfa244-4af4-433f-ac19-668e77e9bd" width="100%" controls></video>


Aplx AI is a lightweight AI assistant built around local and online AI providers, with support for coding, chat, learning, model selection, and developer-focused workflows. (Basically, your personal dock)

## What's new in v1.6

v1.6 is focused on making Aplx more reliable, modular, and useful as a developer assistant.

### Model Switcher

Aplx v1.6 introduces a flexible multi-model orchestration system.

Models can be assigned to three roles:

- **Reasoner** - analyzes the request and creates an implementation plan.
- **Middleman** - converts the plan into precise implementation instructions and reviews the result.
- **Coder** - performs the actual implementation.

The roles are provider-independent, meaning supported providers can be reassigned instead of being permanently tied to a specific vendor.

The pipeline can also validate the final result against the original request.

### Supported switcher providers

The switcher supports configured Aplx providers, including:

- OpenAI
- Anthropic
- Gemini
- OpenAI-compatible providers
- Ollama
- Native Aplx

Provider aliases such as `chatgpt`, `claude`, `local`, and `custom` are normalized automatically.

### Safer project editing

The model switcher can inspect a project, generate file changes, apply them, and then review the actual files written to disk.

Safety protections include:

- Repository-relative file paths
- Path traversal protection
- File deletion blocking
- Validation of generated file-change objects
- Automatic backups of existing files before replacement
- Post-change review based on the files actually written

### Token-Lite

v1.6 includes an improved Token-Lite prompt filter.

Token-Lite is designed to reduce unnecessary prompt text while preserving important content.

It now avoids blindly modifying:

- URLs
- Code blocks
- Important technical tokens
- Protected prompt content

The filter can be enabled from the CLI with:

```bash
python aplx_1.6.py --filter --query "your prompt"
```

The standalone `aplx_filter.py` module contains the TokenFilter implementation.

### Ollama improvements

Ollama handling has been hardened for different installation and package configurations.

Aplx can use the Ollama Python package when available and fall back to the Ollama executable when necessary.

The implementation also handles compatibility with Ollama versions that do or do not support the `think` argument.

### CLI and C++ bridge

v1.6 supports direct command-line queries:

```bash
python aplx_1.6.py --query "Hello"
```

Coding mode:

```bash
python aplx_1.6.py --query "Write a Python calculator" --mode code
```

Token-Lite:

```bash
python aplx_1.6.py --filter --query "Explain this code"
```

A file-based input/output bridge is also included for the Aplx C++ application.

The bridge accepts values such as:

```text
MODE=chat
MODEL=llama3.2:3b
STREAM=1
QUERY=Hello
```

and writes the generated response to the configured output file.

## Core capabilities

- Local AI through Ollama
- Online AI providers
- Multi-provider model selection
- Model-switcher orchestration
- Chat mode
- Coding assistance
- Code review
- Debugging assistance
- Study mode
- Feedback-based learning
- Instant learning
- Autonomous self-teaching
- Self-status and introspection
- Self-upgrade functionality
- Token-Lite prompt filtering
- CLI operation
- C++ file I/O integration

## Requirements

Aplx v1.6 is primarily designed for Python environments.

Depending on which features are used, you may need:

- Python 3
- Ollama
- The Ollama Python package
- Provider API keys for online providers
- The project's native Aplx model files for native-model functionality

Online providers require their corresponding API credentials.

For Ollama, make sure the Ollama executable is installed and available to Aplx, or that the Ollama Python package can connect to the configured server.

## Running Aplx

Interactive mode:

```bash
python aplx_1.6.py
```

Direct query:

```bash
python aplx_1.6.py --query "Hello"
```

Chat mode:

```bash
python aplx_1.6.py --query "Explain recursion" --mode chat
```

Code mode:

```bash
python aplx_1.6.py --query "Create a Python function that sorts a list" --mode code
```

## Model Switcher

The switcher can be configured through Aplx's role configuration functions.

Example role structure:

```text
Reasoner  -> provider/model
Middleman -> provider/model
Coder     -> provider/model
```

Each role can use a different supported provider or model.

The switcher workflow is:

```text
User request
     ↓
Reasoner
     ↓
Middleman / Coordinator
     ↓
Coder
     ↓
Apply changes
     ↓
Middleman / Review
     ↓
Reasoner / Validation
```

This keeps planning, implementation, review, and validation as separate stages.

## Configuration

Provider credentials are read from environment variables where required.

Common examples include:

```text
OPENAI_API_KEY
ANTHROPIC_API_KEY
GEMINI_API_KEY
APLX_OPENAI_COMPAT_BASE_URL
APLX_OPENAI_COMPAT_API_KEY
OLLAMA_HOST
```

Do not hard-code API keys into the source code.

## Native Aplx model

v1.6 contains integration for Aplx's native model/training system.

The native model functionality depends on the corresponding native Aplx model implementation and its training/checkpoint components being available.

If those components are missing, Aplx should report the dependency problem instead of silently pretending that native training is available.

## Self-learning and upgrades

Aplx contains learning-related functionality that can store feedback, instant learnings, knowledge, and queued self-teaching topics.

It also includes proactive upgrade checks and a self-upgrade pipeline.

Self-upgrade functionality should be used carefully because generated source code is involved. v1.6 includes syntax validation and backup behavior before replacing the main source file.

## Project editing safety

When the model switcher applies generated changes:

1. The project path is resolved.
2. Generated paths are checked to remain inside the project.
3. File deletion is blocked.
4. Existing files receive a backup before replacement.
5. Generated content is written.
6. The resulting files are re-read.
7. The post-change project is reviewed.
8. A final validation stage checks the result against the original request.

## Token-Lite module

`aplx_filter.py` provides the standalone TokenFilter implementation.

The filter is intended for reducing prompt overhead, not for changing the meaning of user requests.

It can be imported independently:

```python
from aplx_filter import TokenFilter

filter = TokenFilter()
compressed = filter.compress_prompt("Your prompt here")
```

## Project structure

The v1.6 release centers around:

```text
aplx_1.6.py
aplx_filter.py
```

Additional native-model files and runtime data may be required depending on the enabled features and local setup.

## Notes

Aplx supports both local and online workflows. Availability of a specific provider depends on its credentials, installation, network access, and configured model.

The model switcher is designed to keep provider selection separate from role assignment, allowing the same orchestration architecture to work across different model providers.

## License

Aplx AI is released under the MIT License.

Copyright © Korentic.

See `LICENSE` for the complete license text.

---

**Aplx Dock v1.6**

Built to be flexible.  
Built to run locally.  
Built to keep getting better.

[![GitHub](https://img.shields.io/badge/GitHub-r3nz-00ff88?style=for-the-badge&logo=github)](https://github.com/aplx-renz-sudo)
[![Python](https://img.shields.io/badge/Python-3.9+-00ffff?style=for-the-badge&logo=python)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Active-ff00ff?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-00ff88?style=for-the-badge)](LICENSE)






