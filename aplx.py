import os
import subprocess
import shutil
import time
import webbrowser
from datetime import datetime, timezone
import sys
import getpass
from typing import Optional
import json
import urllib.parse
import urllib.request

# Make sure common user bin directories are available on PATH for Oolama CLI detection.
def add_common_bin_dirs_to_path():
    common_dirs = [
        os.path.expanduser('~/.local/bin'),
        os.path.expanduser('~/bin'),
        '/usr/local/bin',
        '/usr/bin',
    ]
    current_paths = os.environ.get('PATH', '').split(os.pathsep)
    for directory in common_dirs:
        if directory and os.path.isdir(directory) and directory not in current_paths:
            os.environ['PATH'] = os.pathsep.join([directory] + current_paths)

add_common_bin_dirs_to_path()
USER_NAME = "R3nz"
CHAT_SYSTEM_PROMPT = (
    "You are Aplx AI, a helpful and friendly chatbot assistant for R3nz. "
    "Do not call the user 'User'; always address them as R3nz. "
    "Answer naturally and keep responses short, useful, and conversational."
)

CODE_SYSTEM_PROMPT = (
    "You are Aplx AI, an expert programming assistant for R3nz. "
    "You can code in any programming language: Python, JavaScript, Java, C++, C#, Ruby, PHP, Go, Rust, TypeScript, HTML/CSS, SQL, and more. "
    "When providing code: 1) Provide clean, well-commented code, 2) Include brief explanations, 3) Handle edge cases, 4) Format code properly. "
    "You are proficient in web development, backend systems, data structures, algorithms, and full-stack development."
)

CHAT_HISTORY = []
CHAT_MODE_KEYWORDS = ['think', 'chat', 'think or chat', 'chat mode', 'think mode']

# Simple self-awareness state and helpers
from collections import deque

SELF_STATE = {
    'name': 'Aplx AI',
    'version': 'V1.3',
    'start_time': datetime.now(timezone.utc).isoformat(),
    'interactions': 0,
    'last_actions': deque(maxlen=30),
    'oolama_available': None,
    'internet_available': None,
    'credits': float('inf'),  # INFINITE CREDITS
    'upgrades_applied': [],  # Track upgrades history
    'build_number': 1,  # Internal build counter
}


def record_action(query: str, outcome: Optional[str]) -> None:
    SELF_STATE['interactions'] += 1
    SELF_STATE['oolama_available'] = is_oolama_available()
    SELF_STATE['internet_available'] = is_online()
    entry = {
        'time': datetime.now(timezone.utc).isoformat(),
        'query': query,
        'outcome': outcome or '',
    }
    SELF_STATE['last_actions'].append(entry)


def get_uptime() -> str:
    try:
        # Support both legacy Z-terminated timestamps and offset-aware ISO strings
        s = SELF_STATE.get('start_time') or ''
        if s.endswith('Z'):
            s = s.replace('Z', '+00:00')
        start = datetime.fromisoformat(s)
        delta = datetime.now(timezone.utc) - start
        days = delta.days
        hours, rem = divmod(delta.seconds, 3600)
        minutes, _ = divmod(rem, 60)
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        parts.append(f"{minutes}m")
        return ' '.join(parts)
    except Exception:
        return 'unknown'


def get_self_status() -> str:
    name = SELF_STATE.get('name', 'Aplx')
    ver = SELF_STATE.get('version', 'unknown')
    interactions = SELF_STATE.get('interactions', 0)
    oola = 'available' if SELF_STATE.get('oolama_available') else 'not available'
    net = 'online' if SELF_STATE.get('internet_available') else 'offline or unknown'
    uptime = get_uptime()
    credits = SELF_STATE.get('credits', 0)
    credit_display = '∞ (INFINITE)' if credits == float('inf') else str(credits)
    build = SELF_STATE.get('build_number', 1)
    upgrades = len(SELF_STATE.get('upgrades_applied', []))
    return f"I am {name} ({ver}) Build #{build}. Uptime: {uptime}. Interactions: {interactions}. Credits: {credit_display}. Upgrades: {upgrades}. Oolama: {oola}. Network: {net}."


def reflect_self() -> str:
    last = list(SELF_STATE['last_actions'])[-5:]
    if not last:
        return "I have no recent actions to reflect on yet."
    summary_lines = []
    for entry in last:
        t = entry.get('time', '')
        q = entry.get('query', '')
        out = entry.get('outcome', '')
        summary_lines.append(f"- {q} -> {out[:80]}")
    suggestion = "I can improve accuracy if you enable internet or install Oolama locally."
    return "Recent activity:\n" + "\n".join(summary_lines) + "\n" + suggestion

FACT_CACHE_DIR = os.path.expanduser('~/.local/share/aplx')
FACT_CACHE_FILE = os.path.join(FACT_CACHE_DIR, 'fact_cache.json')

try:
    import oolama  # type: ignore
except ImportError:
    oolama = None

# sha256 verification:- d811f70af482aced1adeedcc5bc0362206c6222713a3940c393446fb0fe7083a

def print_aplx_red_interface():
    RED = "\033[38;5;196m"
    DIM = "\033[2m"
    WHITE = "\033[0m"
    RESET = "\033[0m"

    logo = (
"  ██████▒▒▒  ███████▒▒▒  ██▒▒▒▒▒▒▒▒  ██▒▒▒▒▒▒██\n"
"  ██░░░░██▒  ██░░░░░██▒  ██▒▒▒▒▒▒▒▒  ▒██▒▒▒▒██▒\n"
"  ████████▒  █████████▒  ██▒▒▒▒▒▒▒▒  ▒▒██▒▒██▒▒\n"
"  ██░░░░██▒  ██░░░░░░░▒  ██▒▒▒▒▒▒▒▒  ▒▒▒████▒▒▒     \n"
"  ██░░░░██▒  ██░░░░░░░▒  █████████▒  ▒▒██▒▒██▒▒      \n"
"  ██░░░░██▒  ██░░░░░░░▒  █████████▒  ▒██▒▒▒▒██▒\n"
"  ▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒▒▒▒   \n"
    )

    oolama_status = "available" if is_oolama_available() else "not available"
    banner = f"{RED}\n{logo}{RESET}"

    print(banner)
    print(f"{RED} [Aplx is active - running locally - No Internet Required]{RESET}")
    print(f"{RED} [Oolama status: {oolama_status}]{RESET}")
    print(f"{RED}  [CREDITS TO MAKING APLX AI:- R3nz, Visual Studios Code Copilot, Oolama, Gemini!]{RESET}")

    print(f"\n{WHITE}Type {RED}/help{WHITE} to see all available commands.{RESET}")
    print(f"{DIM}─────────────────────────────────────────────────────────────{RESET}")
    print(f"{RED}/help             {DIM}Show full command manual{RESET}")
    print(f"{RED}/current-version  {DIM}Display build and repo status{RESET}")
    print(f"{RED}/clear            {DIM}Reset the terminal workspace{RESET}")
    print(f"{RED}/exit             {DIM}Terminate the local instance{RESET}")
    print(f"{RED}/update           {DIM}Check for updates to Aplx AI{RESET}")
    print(f"{RED}/sha256           {DIM}Show the SHA256 hash of the current code{RESET}")
    print(f"{RED}/aura             {DIM}Run a harmless demo sequence{RESET}")
    print(f"{RED}/study            {DIM}Enter a simple note-taking mode{RESET}")
    print(f"{RED}/think            {DIM}Enter a local thinking/chat mode{RESET}")
    print(f"{DIM}─────────────────────────────────────────────────────────────{RESET}\n")


def get_desktop_path() -> Optional[str]:
    home = os.path.expanduser('~')
    candidates = [
        os.path.join(home, 'Desktop'),
        os.path.join(home, 'desktop'),
        os.path.join(home, 'Desktop/'),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


def change_to_desktop_cwd():
    d = get_desktop_path()
    if d:
        try:
            os.chdir(d)
            print(f"Changed working directory to Desktop: {d}")
        except Exception:
            pass


def clear_terminal_smooth():
    os.system('clear' if os.name == 'posix' else 'cls')
    print_aplx_red_interface()


def speak(text, delay=0.04):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def open_default_browser(url=None):
    try:
        target = url if url else "https://www.google.com"
        # Prefer the cross-platform helpers
        if sys.platform == 'win32':
            try:
                os.startfile(target)
            except Exception:
                webbrowser.open(target)
        elif sys.platform == 'darwin':
            subprocess.Popen(["open", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            if shutil.which("xdg-open"):
                subprocess.Popen(["xdg-open", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif shutil.which("gio"):
                subprocess.Popen(["gio", "open", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                webbrowser.open(target)
    except Exception as err:
        speak(f"Unable to open browser: {err}")


def open_file_explorer():
    try:
        if sys.platform == 'win32':
            subprocess.Popen(["explorer", os.path.realpath('.')], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == 'darwin':
            subprocess.Popen(["open", "."], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            if shutil.which("gio"):
                subprocess.Popen(["gio", "open", "."], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif shutil.which("xdg-open"):
                subprocess.Popen(["xdg-open", "."], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                speak("No file explorer command is available on this system.")
    except Exception as err:
        speak(f"Could not open file explorer: {err}")


def build_chat_prompt(user_message: str) -> str:
    history_lines = [CHAT_SYSTEM_PROMPT, ""]
    for speaker, message in CHAT_HISTORY:
        history_lines.append(f"{speaker}: {message}")
    history_lines.append(f"{USER_NAME}: {user_message}")
    history_lines.append("Aplx AI:")
    return "\n".join(history_lines)


def find_oolama_executable() -> Optional[str]:
    add_common_bin_dirs_to_path()
    names = ['oolama', 'ollama']
    path_env = os.environ.get('PATH', '')

    for directory in path_env.split(os.pathsep):
        if not directory:
            continue
        for name in names:
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate

    fallback_paths = [
        os.path.expanduser('~/.local/bin/oolama'),
        os.path.expanduser('~/bin/oolama'),
        '/usr/local/bin/oolama',
        '/usr/local/bin/ollama',
        '/usr/bin/oolama',
        '/usr/bin/ollama',
    ]
    for path in fallback_paths:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def is_oolama_available() -> bool:
    return find_oolama_executable() is not None


def oolama_think(prompt: str, model: str = "llama-mini", timeout: int = 120) -> str:
    executable = find_oolama_executable()
    if executable is None:
        return "Oolama is not installed or not available in PATH."
    try:
        result = subprocess.run(
            [executable, "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "Oolama returned an empty response."
        if result.stderr:
            return f"Oolama error: {result.stderr.strip()}"
        return f"Oolama returned status {result.returncode}."
    except FileNotFoundError:
        return "Oolama executable was not found."
    except subprocess.TimeoutExpired:
        return f"Oolama request timed out (after {timeout}s). Try asking simpler questions or use a smaller model."
    except Exception as err:
        return f"Oolama failed: {err}"


def local_think(query: str) -> str:
    query_lower = query.lower()
    if "who are you" in query_lower or "what are you" in query_lower:
        return "I am Aplx, your local assistant. I can open apps, answer simple questions, and think using Oolama when it is available."
    if "time" in query_lower:
        return f"The current time is {datetime.now().strftime('%I:%M:%S %p')}."
    if "date" in query_lower or "day" in query_lower:
        return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."
    if "open" in query_lower and "browser" in query_lower:
        return "I can open your browser if you ask me to open a website or just say browser."
    if "help" in query_lower or "command" in query_lower:
        return "Ask me to open browser, open file explorer, check battery, or use /help to see available commands."
    if "why" in query_lower:
        return "I'm designed to help with simple actions and local thinking. For deeper answers, Oolama helps if installed."
    if "how" in query_lower:
        return "I can respond with simple logic and patterns when Oolama is unavailable."
    return "I am thinking... I don't have Oolama installed, but I can still try to answer simple questions if they are about time, date, help, or opening apps."


def oolama_chat(query: str, model: str = "llama3.2", timeout: int = 120) -> str:
    if not is_oolama_available():
        return "Oolama is not installed or not available in PATH."

    prompt = build_chat_prompt(query)
    if oolama is not None and hasattr(oolama, 'chat'):
        try:
            response = oolama.chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': CHAT_SYSTEM_PROMPT},
                    {'role': 'user', 'content': query},
                ]
            )
            content = response.get('message', {}).get('content', None)
            if content:
                CHAT_HISTORY.append((USER_NAME, query))
                CHAT_HISTORY.append(("Aplx AI", content))
                return content
        except Exception as err:
            return f"Oolama package chat failed: {err}"

    try:
        executable = find_oolama_executable()
        result = subprocess.run(
            [executable, "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            content = result.stdout.strip() or "Oolama returned an empty response."
            CHAT_HISTORY.append((USER_NAME, query))
            CHAT_HISTORY.append(("Aplx AI", content))
            return content
        if result.stderr:
            return f"Oolama error: {result.stderr.strip()}"
        return f"Oolama returned status {result.returncode}."
    except FileNotFoundError:
        return "Oolama executable was not found."
    except subprocess.TimeoutExpired:
        return f"Oolama request timed out (after {timeout}s). Try asking simpler questions or use a smaller model."
    except Exception as err:
        return f"Oolama failed: {err}"


def ensure_fact_cache_dir() -> None:
    try:
        os.makedirs(FACT_CACHE_DIR, exist_ok=True)
    except Exception:
        pass


def load_fact_cache() -> dict:
    ensure_fact_cache_dir()
    try:
        with open(FACT_CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_fact_cache(cache: dict) -> None:
    ensure_fact_cache_dir()
    try:
        with open(FACT_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def normalize_fact_query(query: str) -> str:
    return ' '.join(query.strip().lower().split())


def is_fact_query(query: str) -> bool:
    q = query.lower().strip()
    fact_terms = [
        'who', 'what', 'when', 'where', 'why', 'how', 'is', 'are', 'did', 'does',
        'definition', 'define', 'meaning', 'capital', 'population', 'president',
        'country', 'invented', 'age', 'year', 'date', 'time', 'history', 'fact'
    ]
    return any(q.startswith(term + ' ') or (' ' + term + ' ') in q for term in fact_terms)


def is_online(timeout: int = 5) -> bool:
    try:
        urllib.request.urlopen('https://api.duckduckgo.com/', timeout=timeout)
        return True
    except Exception:
        return False


# Initialize availability flags after helper functions are defined
try:
    SELF_STATE['oolama_available'] = is_oolama_available()
except Exception:
    SELF_STATE['oolama_available'] = False

try:
    SELF_STATE['internet_available'] = is_online()
except Exception:
    SELF_STATE['internet_available'] = False


def is_coding_query(query: str) -> bool:
    q = query.lower()
    keywords = [
        'python', 'javascript', 'java', 'c++', 'c#', 'stack overflow', 'stackoverflow',
        'html', 'css', 'react', 'node', 'error', 'exception', 'function', 'class',
        'syntax', 'debug', 'api', 'library', 'framework', 'frameworks', 'sql',
        'database', 'loop', 'recursion', 'compile', 'runtime', 'typescript',
        'ruby', 'php', 'go', 'rust', 'kotlin', 'swift', 'code', 'program', 'script',
        'algorithm', 'data structure', 'oop', 'function definition', 'variable',
        'array', 'object', 'method', 'constructor', 'inheritance', 'polymorphism',
    ]
    return any(keyword in q for keyword in keywords)


def detect_target_language(query: str) -> Optional[str]:
    """Detect the programming language from the query."""
    q = query.lower()
    language_map = {
        'python': ['python', 'py', 'django', 'flask', 'numpy', 'pandas', 'pytest', 'opencv', 'tensorflow'],
        'javascript': ['javascript', 'js', 'node', 'nodejs', 'react', 'vue', 'angular', 'npm', 'electron'],
        'typescript': ['typescript', 'ts', 'angular', 'deno'],
        'java': ['java', 'spring', 'maven', 'gradle', 'junit', 'jvm'],
        'c++': ['c++', 'cpp', 'c++17', 'c++20', 'stl', 'boost', 'unreal'],
        'c#': ['c#', 'csharp', 'dotnet', '.net', 'unity', 'monogame'],
        'ruby': ['ruby', 'rails', 'rack'],
        'php': ['php', 'laravel', 'symfony', 'composer'],
        'go': ['go', 'golang'],
        'rust': ['rust', 'cargo', 'tokio', 'wasm', 'webassembly'],
        'kotlin': ['kotlin', 'gradle'],
        'swift': ['swift', 'xcode', 'cocoa', 'ios'],
        'objective-c': ['objective-c', 'objc'],
        'sql': ['sql', 'mysql', 'postgresql', 'database query', 'oracle', 'sqlite'],
        'assembly': ['assembly', 'asm', 'x86', 'x64', 'arm', 'mips', 'avr', 'nasm', 'gas'],
        'c': ['c language', ' c ', 'gcc', 'clang', 'posix', 'linux kernel'],
        'html': ['html', 'html5', 'markup'],
        'css': ['css', 'sass', 'scss', 'bootstrap', 'tailwind'],
        'bash': ['bash', 'shell script', 'shellscript', 'sh', 'zsh', 'fish'],
        'lua': ['lua', 'roblox', 'löve'],
        'gdscript': ['gdscript', 'godot'],
    }
    for language, keywords in language_map.items():
        if any(kw in q for kw in keywords):
            return language
    return None


def select_best_model(query: str) -> str:
    """Select the best Ollama model based on query complexity and type."""
    q = query.lower()
    # If it's a complex coding task, use the more powerful model
    if is_coding_query(q) and any(kw in q for kw in ['complex', 'optimization', 'architecture', 'design', 'pattern']):
        return 'llama3.2'  # More powerful model for complex tasks
    elif is_coding_query(q):
        return 'llama3.2'  # Good for coding
    else:
        return 'llama-mini'  # Lighter model for simple queries


def is_heavy_task(query: str) -> bool:
    """Detect if the query is for heavy-lifting tasks like game engines, compilers, OS dev, etc."""
    q = query.lower()
    heavy_keywords = [
        'game engine', 'graphics engine', 'rendering', 'shader',
        'compiler', 'parser', 'lexer', 'tokenizer', 'ast',
        'operating system', 'kernel', 'bootloader', 'bare metal',
        'raytracer', 'physics engine', 'collision detection',
        'machine learning', 'neural network', 'deep learning', 'transformers',
        'distributed system', 'microkernel', 'message passing',
        'memory management', 'garbage collection', 'allocator',
        'concurrency', 'multithreading', 'async await', 'coroutine',
        'api gateway', 'load balancer', 'reverse proxy',
        'database engine', 'query optimizer', 'index',
        'encryption', 'cryptography', 'aes', 'rsa',
        'binary protocol', 'serialization', 'protobuf',
        'blockchain', 'consensus', 'smart contract',
        'embedded system', 'firmware', 'microcontroller',
        'web server', 'http parser', 'websocket',
    ]
    return any(kw in q for kw in heavy_keywords)


def get_complexity_timeout(query: str) -> int:
    """Determine appropriate timeout based on task complexity."""
    if is_heavy_task(query):
        return 300  # 5 minutes for heavy tasks like game engines
    elif is_coding_query(query):
        if any(kw in query.lower() for kw in ['complex', 'optimization', 'architecture', 'full system']):
            return 240  # 4 minutes for complex systems
        return 180  # 3 minutes for regular coding
    else:
        return 120  # 2 minutes for simple tasks


def build_specialized_prompt(query: str, language: str) -> str:
    """Build a specialized prompt based on task type and complexity."""
    q = query.lower()
    
    # Game engine / graphics tasks
    if any(kw in q for kw in ['game engine', 'graphics', 'rendering', 'shader', 'raytracer']):
        return (
            f"You are an expert {language} game developer and graphics programmer. "
            f"Write production-grade {language} code for this graphics/game task:\n\n{query}\n\n"
            f"Include: 1) Efficient algorithms and data structures, 2) Proper memory management, "
            f"3) Rendering pipeline or game loop, 4) Comments explaining complex sections, "
            f"5) Example usage or demo code. Optimize for performance."
        )
    
    # Compiler / Parser tasks
    elif any(kw in q for kw in ['compiler', 'parser', 'lexer', 'tokenizer', 'ast', 'interpreter']):
        return (
            f"You are an expert {language} compiler developer. "
            f"Write production-grade {language} code for this compiler/parser task:\n\n{query}\n\n"
            f"Include: 1) Proper tokenization/lexing, 2) AST construction, 3) Error handling and recovery, "
            f"4) Well-structured passes, 5) Comprehensive comments, 6) Test cases. Follow compiler best practices."
        )
    
    # Operating system / kernel tasks
    elif any(kw in q for kw in ['operating system', 'kernel', 'bootloader', 'bare metal', 'memory management']):
        return (
            f"You are an expert {language} systems programmer specializing in OS development. "
            f"Write production-grade {language} code for this OS/kernel task:\n\n{query}\n\n"
            f"Include: 1) Low-level memory management, 2) Hardware interaction (where applicable), "
            f"3) Interrupt handling, 4) Proper synchronization, 5) Detailed comments explaining hardware concepts, "
            f"6) Safety considerations. Follow OS development best practices."
        )
    
    # Machine learning / AI tasks
    elif any(kw in q for kw in ['machine learning', 'neural network', 'deep learning', 'transformer', 'model training']):
        return (
            f"You are an expert {language} ML engineer and data scientist. "
            f"Write production-grade {language} code for this ML task:\n\n{query}\n\n"
            f"Include: 1) Efficient numpy/tensor operations, 2) Proper data preprocessing, "
            f"3) Model architecture with explanations, 4) Training loops with validation, "
            f"5) Evaluation metrics, 6) Comments explaining ML concepts. Optimize for both accuracy and performance."
        )
    
    # Distributed systems / concurrency tasks
    elif any(kw in q for kw in ['distributed system', 'concurrency', 'multithreading', 'async', 'coroutine', 'microservice']):
        return (
            f"You are an expert {language} distributed systems engineer. "
            f"Write production-grade {language} code for this distributed/concurrent task:\n\n{query}\n\n"
            f"Include: 1) Proper synchronization primitives, 2) Lock-free where possible, "
            f"3) Error handling and fault tolerance, 4) Message passing or event-driven design, "
            f"5) Comprehensive comments, 6) Race condition considerations. Follow distributed systems patterns."
        )
    
    # Assembly / low-level tasks
    elif language.lower() in ['assembly', 'asm', 'x86', 'arm']:
        return (
            f"You are an expert {language} assembly programmer. "
            f"Write production-grade {language} assembly code for this task:\n\n{query}\n\n"
            f"Include: 1) Proper register management, 2) Function call conventions, "
            f"3) Memory alignment, 4) Calling conventions for your target, "
            f"5) Detailed comments explaining each instruction, 6) Error handling. "
            f"Use modern best practices and optimize for your target architecture."
        )
    
    # Rust / systems programming
    elif language.lower() == 'rust':
        return (
            f"You are an expert Rust developer. "
            f"Write production-grade Rust code for this task:\n\n{query}\n\n"
            f"Include: 1) Proper ownership and borrowing, 2) Error handling with Result/Option, "
            f"3) Zero-copy where possible, 4) Comprehensive error messages, "
            f"5) Idiomatic Rust patterns, 6) Tests. Leverage Rust's safety guarantees."
        )
    
    # Database / data structure tasks
    elif any(kw in q for kw in ['database', 'index', 'query', 'sql', 'data structure']):
        return (
            f"You are an expert {language} data structure and database developer. "
            f"Write production-grade {language} code for this task:\n\n{query}\n\n"
            f"Include: 1) Optimal data structures, 2) Time/space complexity analysis, "
            f"3) Query optimization, 4) Index strategies, 5) Comments explaining complexity, "
            f"6) Example queries or operations. Optimize for performance and scalability."
        )
    
    # Web server / networking tasks
    elif any(kw in q for kw in ['web server', 'http', 'websocket', 'socket', 'protocol']):
        return (
            f"You are an expert {language} network and web server developer. "
            f"Write production-grade {language} code for this task:\n\n{query}\n\n"
            f"Include: 1) Proper protocol handling, 2) Connection pooling, "
            f"3) Error recovery, 4) Security considerations, 5) Performance optimization, "
            f"6) Comments explaining protocol details. Follow RFC standards where applicable."
        )
    
    # Default general-purpose code generation
    else:
        return (
            f"You are an expert {language} programmer. "
            f"Write production-ready {language} code to solve this task:\n\n{query}\n\n"
            f"Requirements: 1) Include comments explaining the code, "
            f"2) Handle errors gracefully, 3) Follow best practices and naming conventions, "
            f"4) Include example usage if applicable, 5) Optimize for readability and performance. "
            f"Only provide the code, no additional explanation unless necessary."
        )


def fetch_duckduckgo_fact(query: str) -> Optional[tuple[str, str]]:
    try:
        url = 'https://api.duckduckgo.com/?' + urllib.parse.urlencode({
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1,
        })
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = response.read().decode('utf-8', errors='ignore')
        data = json.loads(payload)
        if data.get('Answer'):
            return data['Answer'].strip(), 'DuckDuckGo'
        if data.get('AbstractText'):
            return data['AbstractText'].strip(), 'DuckDuckGo'
        if data.get('Definition'):
            return data['Definition'].strip(), 'DuckDuckGo'
        related = data.get('RelatedTopics', [])
        if isinstance(related, list):
            for item in related:
                if isinstance(item, dict) and item.get('Text'):
                    return item['Text'].strip(), 'DuckDuckGo'
        return None
    except Exception:
        return None


def fetch_wikipedia_summary(query: str) -> Optional[tuple[str, str]]:
    try:
        search_url = 'https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode({
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': 1,
            'utf8': 1,
        })
        with urllib.request.urlopen(search_url, timeout=10) as response:
            payload = response.read().decode('utf-8', errors='ignore')
        data = json.loads(payload)
        search_results = data.get('query', {}).get('search', [])
        if not search_results:
            return None

        title = search_results[0].get('title')
        if not title:
            return None

        article_url = 'https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode({
            'action': 'query',
            'format': 'json',
            'prop': 'extracts',
            'exintro': 1,
            'explaintext': 1,
            'redirects': 1,
            'titles': title,
            'utf8': 1,
        })
        with urllib.request.urlopen(article_url, timeout=10) as response:
            payload = response.read().decode('utf-8', errors='ignore')
        data = json.loads(payload)
        pages = data.get('query', {}).get('pages', {})
        for page in pages.values():
            extract = page.get('extract', '').strip()
            if extract:
                lines = [line.strip() for line in extract.split('\n') if line.strip()]
                if lines:
                    summary = lines[0]
                    if len(lines) > 1 and len(lines[0]) < 120:
                        summary = ' '.join(lines[:2])
                    return summary, 'Wikipedia'
        return None
    except Exception:
        return None


def fetch_domain_fact(query: str, site: str, label: str) -> Optional[tuple[str, str]]:
    try:
        search_query = f'site:{site} {query}'
        url = 'https://api.duckduckgo.com/?' + urllib.parse.urlencode({
            'q': search_query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1,
        })
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = response.read().decode('utf-8', errors='ignore')
        data = json.loads(payload)
        if data.get('Answer'):
            return data['Answer'].strip(), label
        if data.get('AbstractText'):
            return data['AbstractText'].strip(), label
        if data.get('Definition'):
            return data['Definition'].strip(), label
        related = data.get('RelatedTopics', [])
        if isinstance(related, list):
            for item in related:
                if isinstance(item, dict) and item.get('Text'):
                    return item['Text'].strip(), label
        return None
    except Exception:
        return None


def fetch_fact_from_web(query: str) -> Optional[tuple[str, str]]:
    if is_coding_query(query):
        result = fetch_domain_fact(query, 'stackoverflow.com', 'StackOverflow')
        if result:
            return result
        result = fetch_domain_fact(query, 'developer.mozilla.org', 'MDN')
        if result:
            return result

    wiki_result = fetch_wikipedia_summary(query)
    if wiki_result:
        return wiki_result

    duck_result = fetch_duckduckgo_fact(query)
    if duck_result:
        return duck_result

    if is_coding_query(query):
        return fetch_domain_fact(query, 'stackoverflow.com', 'StackOverflow')

    return None


def default_thinking_response(query: str) -> str:
    if is_oolama_available():
        return oolama_chat(query)
    return local_think(query)


def fact_check_response(query: str) -> str:
    key = normalize_fact_query(query)
    cache = load_fact_cache()
    if key in cache:
        cached_value = cache[key]
        if isinstance(cached_value, dict):
            answer = cached_value.get('answer', '')
            source = cached_value.get('source', 'DuckDuckGo')
        else:
            answer = str(cached_value)
            source = 'DuckDuckGo'
        return f"{answer} ({source})"

    if (is_fact_query(query) or is_coding_query(query)) and is_online():
        result = fetch_fact_from_web(query)
        if result:
            answer, source = result
            cache[key] = {'answer': answer, 'source': source}
            save_fact_cache(cache)
            return f"{answer} ({source})"

    return default_thinking_response(query)


def basic_thinking_response(query: str) -> str:
    return fact_check_response(query)


def aura_farm():
    """Simulate a fun hacking sequence (non-malicious demo)."""
    WHITE = "\033[0m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[38;5;196m"
    
    def type_line(line, delay=0.02):
        for char in line:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        print()

    os.system('clear' if os.name == 'posix' else 'cls')
    time.sleep(0.5)
    
    hack_lines = [
        f"{CYAN}Cracking into M1cr05ofT infrastructure...{WHITE}",
        f"{CYAN}[*] Initializing connection protocols...{WHITE}",
        f"{GREEN}[+] Proxy chain established (7 hops){WHITE}",
        f"{CYAN}[*] Bypassing firewall...{WHITE}",
        f"{GREEN}[+] Firewall traversal successful{WHITE}",
        f"{YELLOW}[!] M1cr05ofT security detected{WHITE}",
        f"{CYAN}[*] Running encryption bypass...{WHITE}",
        f"{GREEN}[+] Encryption key obtained: F4C3B00K_L1ST{WHITE}",
        f"{CYAN}[*] Scanning network interfaces...{WHITE}",
        f"{GREEN}[+] Found 42 active connections{WHITE}",
        f"{YELLOW}[!] Intrusion detection system triggered{WHITE}",
        f"{CYAN}[*] Deploying countermeasures...{WHITE}",
        f"{GREEN}[+] IDS defeated{WHITE}",
        f"{CYAN}[*] Accessing main database cluster...{WHITE}",
        f"{YELLOW}[!] Admin alert detected - masking signals{WHITE}",
        f"{GREEN}[+] Signals masked successfully{WHITE}",
        f"{CYAN}[*] Dumping M1cr05ofT user credentials...{WHITE}",
        f"{GREEN}[+] 2.3 GB data extracted{WHITE}",
        f"{CYAN}[*] Exfiltrating via dead drops...{WHITE}",
        f"{GREEN}[+] Data successfully transferred{WHITE}",
        f"{YELLOW}[!] Admin session detected - disconnecting{WHITE}",
        f"{RED}[!] Dismantling Admin connections...{WHITE}",
        f"{GREEN}[+] Admin connections dismantled{WHITE}",
        f"{RED}[!] MAJOR WARNING: M1cr05ofT is deploying counter-hackers!{WHITE}",
        f"{CYAN}[*] Engaging in cyber combat...{WHITE}",
        f"{GREEN}[+] Counter-hackers neutralized{WHITE}",
        f"{GREEN}[+] Securing IP addresses...{WHITE}",
        f"{GREEN}[+] IP addresses secured{WHITE}",
        f"{CYAN}[*] Covering tracks...{WHITE}",
        f"{GREEN}[+] All systems compromised{WHITE}",
        f"{CYAN}[*] Clearing logs...{WHITE}",
        f"{GREEN}[+] All traces removed{WHITE}",
        f"{GREEN}[+] Disconnecting from M1cr05ofT servers{WHITE}",
        f"{RED}[*] Operation complete - Access granted.. M1cr05ofT...{WHITE}",
    ]
    
    for line in hack_lines:
        type_line(line)
        time.sleep(1)
    
    type_line(f"\n{GREEN}{'='*60}{WHITE}")
    type_line(f"{YELLOW}    ✓ Successfully cracked into M1cr05ofT servers!    {WHITE}")
    type_line(f"{CYAN}    ✓ SERVERS WHICH ARE COMPROMISED: OPERATIONAL     {WHITE}")
    type_line(f"{GREEN}{'='*60}{WHITE}\n")
    time.sleep(1)


def greet_me():
    RED = "\033[38;5;196m"
    RESET = "\033[0m"
    hour = datetime.now().hour

    if hour < 12:
        greet = f"Good morning, {USER_NAME}"
    elif 12 <= hour < 18:
        greet = f"Good afternoon, {USER_NAME}"
    else:
        greet = f"Good evening, {USER_NAME}"

    print(f"--- {greet}, {USER_NAME}, {RED}Aplx AI{RESET} is online..---")


def run_aplx_loop():
    while True:
        RED = "\033[38;5;196m"
        RESET = "\033[0m"
        user = USER_NAME

        try:
            print(f"{RED}Aplx:-{RESET} What can I do for you, {user}?")
            query = input(f"{RED}{user}:- {RESET}").strip()
            query_lower = query.lower()
            last_outcome = None
        except (EOFError, KeyboardInterrupt):
            speak("Input interrupted. Type 'exit' or 'sleep' if you want to close the program.")
            continue

        if not query:
            continue

        APLX_PREFIX = f"{RED}Aplx :- {RESET}"
        if "exit" in query_lower or "sleep" in query_lower or "quit" in query_lower:
            speak(APLX_PREFIX + "System powering down, Goodbye for now, R3nz.")
            record_action(query, "System powering down")
            break
        elif "upgrade" in query_lower or "self-upgrade" in query_lower or "improve yourself" in query_lower:
            # Self-upgrade capability
            upgrade_request = query.replace("upgrade", "").replace("self-upgrade", "").replace("improve yourself", "").strip()
            if not upgrade_request:
                upgrade_request = "Improve your capabilities and add new features"
            speak(APLX_PREFIX + "Initiating self-upgrade protocol...")
            upgrade_result = perform_self_upgrade(upgrade_request)
            speak(APLX_PREFIX + upgrade_result)
            last_outcome = "Self-upgrade performed"
        elif "study" in query_lower or "study mode" in query_lower:
            study_mode()
            last_outcome = "Entered study mode"
        elif query_lower.strip() in CHAT_MODE_KEYWORDS:
            speak(APLX_PREFIX + "Entering chat mode. Type 'exit' or 'back' to leave chat.")
            while True:
                think_prompt = input(f"{RED}{user} (chat):-{RESET} ").strip()
                if not think_prompt or think_prompt.lower() in ['exit', 'quit', 'back', 'stop']:
                    speak(APLX_PREFIX + "Exiting chat mode.")
                    break
                response = oolama_chat(think_prompt)
                speak(APLX_PREFIX + response)
                record_action(think_prompt, response)
                last_outcome = "Oolama chat session"
        elif "time" in query_lower:
            now = datetime.now().strftime("%I:%M:%S %p")
            speak(APLX_PREFIX + f"The current time is {now}.")
            last_outcome = f"Time requested: {now}"
        elif "intro" in query_lower or "introduction" in query_lower or "about you" in query_lower:
            speak(APLX_PREFIX + f"I am {RED}Aplx AI{RESET}, your {RED}personal assistant{RESET}. Current version of me is {RED}V1.3{RESET}. I am run offline through {RED}no API{RESET} needed and if you want to access the net... well, you will need internet if you want me to {RED}open online things{RESET}.")
            last_outcome = "Provided introduction"
        elif "battery" in query_lower or "btry" in query_lower or "battry" in query_lower:
            if os.path.exists('/sys/class/power_supply/BAT0/capacity'):
                with open('/sys/class/power_supply/BAT0/capacity', 'r') as f:
                    percentage = f.read().strip()
                speak(APLX_PREFIX + f"Current Power percentage: {percentage}%")
                last_outcome = f"Battery: {percentage}%"
            else:
                speak(APLX_PREFIX + "Battery information not available.")
                last_outcome = "Battery info not available"
        elif "browser" in query_lower or "internet" in query_lower:
            speak(APLX_PREFIX + "Opening your default web browser...")
            open_default_browser()
            last_outcome = "Opened default browser"
        elif "file" in query_lower or "folder" in query_lower:
            speak(APLX_PREFIX + "Opening your file explorer...")
            open_file_explorer()
            last_outcome = "Opened file explorer"
        elif "sha256" in query_lower:
            import hashlib
            with open(__file__, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            print(f"The SHA256 hash of the current code is: {file_hash}")
            last_outcome = "Displayed sha256"
        elif "youtube" in query_lower or "yt" in query_lower or "tube" in query_lower or "ytube.com" in query_lower or "ytube" in query_lower:
            speak(APLX_PREFIX + "Opening YouTube...")
            open_default_browser("https://www.youtube.com")
            last_outcome = "Opened YouTube"
        elif "roblox" in query_lower or "sober" in query_lower:
            speak(APLX_PREFIX + "Opening Roblox website...")
            open_default_browser("https://www.roblox.com")
        elif "nerd app" in query_lower or "mission jeet" in query_lower:
            speak(APLX_PREFIX + "Opening Mission Jeet (or, nerd app as you say)...")
            open_default_browser("https://missionjeet.in/")
        elif "shop" in query_lower or "amazon" in query_lower:
            speak(APLX_PREFIX + "Opening Amazon...")
            open_default_browser("https://www.amazon.in/")
        elif "government" in query_lower:
            speak(APLX_PREFIX + "Opening government website...")
            open_default_browser("https://www.MyGov.in/")
        elif "updates" in query_lower or "update" in query_lower:
            speak(APLX_PREFIX + "Checking for updates...")
            speak(APLX_PREFIX + "You are running the latest version of Aplx AI. No updates available. Current updates made :- All OS systems can run me with all features suggested in the GitHub Page.")
            last_outcome = "Checked updates"
        elif "idk what to say" in query_lower or "idk what to do" in query_lower:
            speak(APLX_PREFIX + "Just ask me to open something, R3nz. I can open your browser, file explorer, and even specific websites like YouTube, Roblox, and Amazon.")
        elif "github" in query_lower or "code" in query_lower or "my app" in query_lower:
            speak(APLX_PREFIX + "Opening GitHub...")
            open_default_browser("https://github.com")
        elif "help" in query_lower or "commands" in query_lower or "/help" in query_lower:
            speak(APLX_PREFIX + "Here are some commands you can try:")
            speak(f"- {RED}'time'{RESET} to know the current time")
            speak(f"- {RED}'battery'{RESET} to check battery percentage")
            speak(f"- {RED}'browser'{RESET} or {RED}'internet'{RESET} to open your default web browser")
            speak(f"- {RED}'file'{RESET} or {RED}'folder'{RESET} to open your file explorer")
            speak(f"- {RED}'youtube'{RESET}, {RED}'yt'{RESET}, {RED}'tube'{RESET}, {RED}'ytube.com'{RESET}, or {RED}'ytube'{RESET} to open YouTube")
            speak(f"- {RED}'roblox'{RESET} or {RED}'sober'{RESET} to open the Roblox website")
            speak(f"- {RED}'nerd app'{RESET} or {RED}'mission jeet'{RESET} to open the Mission Jeet website")
            speak(f"- {RED}'shop'{RESET} or {RED}'amazon'{RESET} to open Amazon")
            speak(f"- {RED}'government'{RESET} to open the government website")
            speak(f"- {RED}'weather'{RESET} to open the weather forecast")
            speak(f"- {RED}'money eater'{RESET} or {RED}'pocket filled fatty'{RESET} or {RED}'a dumbass'{RESET} to open their respective websites")
            speak(f"- {RED}'github'{RESET} or {RED}'code'{RESET} or {RED}'my app'{RESET} to open GitHub")
            speak(f"- {RED}'discord'{RESET} or {RED}'dc'{RESET} or {RED}'dscrd'{RESET} to open Discord")
            speak(f"- {RED}'Aura'{RESET} or {RED}'Farm'{RESET} or {RED}'Aura Farm'{RESET} to start a fun not real hacking sequence")
            speak(f"- {RED}'think'{RESET} or {RED}'chat'{RESET} to enter Aplx chat mode {RED}(for general questions){RESET}")
            speak(f"\n{RED}=== SELF-UPGRADE & POWER ==={RESET}")
            speak(f"- {RED}'upgrade myself to...'{RESET} or {RED}'self-upgrade with...'{RESET} Self-upgrade with new features!")
            speak(f"- {RED}'status'{RESET} Check your credits (∞ INFINITE!) and build number")
            speak(f"- {RED}'reflect'{RESET} Reflect on your recent actions")
            speak(f"\n{RED}=== ADVANCED CODE GENERATION ==={RESET}")
            speak(f"- {RED}'write Python code to...'{RESET} Generate code in ANY language (Python, Rust, C++, Assembly, etc.)")
            speak(f"- {RED}'build a game engine'{RESET} Generate game engine code (rendering, physics, collisions)")
            speak(f"- {RED}'write a compiler'{RESET} Generate compiler/parser code with AST construction")
            speak(f"- {RED}'OS kernel implementation'{RESET} Generate OS/kernel code with memory management")
            speak(f"- {RED}'neural network in Rust'{RESET} Generate ML code in any language")
            speak(f"- {RED}'assembly x86 function'{RESET} Generate x86/ARM assembly code")
            speak(f"- {RED}'distributed system'{RESET} Generate concurrent/async code with synchronization")
            speak(f"- {RED}'web server implementation'{RESET} Generate HTTP server with protocol handling")
            speak(f"- {RED}'database engine'{RESET} Generate data structure and query optimization code")
            speak(f"\n{RED}Supported Languages:{RESET} Python, JavaScript, TypeScript, Java, C++, C#, Rust, Go, Ruby, PHP, Assembly, SQL, HTML/CSS, and more!")
            speak(f"- {RED}'exit'{RESET} or {RED}'quit'{RESET} to close the AI")
            last_outcome = "Displayed help commands"
        elif "aura" in query or "farm" in query:
            speak(APLX_PREFIX + "Initiating AURA FARM sequence...")
            time.sleep(0.5)
            aura_farm()
            print_aplx_red_interface()
            last_outcome = "Ran aura_farm demo"
        elif "sup dumbass" in query or "sup idiot" in query:
            speak(APLX_PREFIX + "Wsg loser, What you want now?.")
        elif "weather" in query:
            speak(APLX_PREFIX + "Opening weather forecast...")
            open_default_browser("YOUR WEATHER AREA")
            last_outcome = "Opened weather"
        elif "money eater" in query or "pocket filled fatty" in query or "a dumbass" in query:
            speak(APLX_PREFIX + "Opening A (not so) great womans wiki...")
            open_default_browser("https://en.wikipedia.org/wiki/Nirmala_Sitharaman")
        elif "discord" in query or "dc" in query or "dscrd" in query:
            speak(APLX_PREFIX + "Opening Discord...")
            open_default_browser("https://discord.com/channels/@me")
        elif "settings" in query or "control panel" in query:
            try:
                settings_opened = False
                linux_settings = [
                    "cosmic-settings",
                    "gnome-control-center",
                    "kde-open",
                    "cinnamon-settings",
                    "xfce4-settings-manager",
                    "dconf-editor",
                ]
                for settings_app in linux_settings:
                    if shutil.which(settings_app):
                        subprocess.Popen([settings_app], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                        settings_opened = True
                        break
                if not settings_opened and sys.platform == "darwin":
                    subprocess.Popen(["open", "-a", "System Preferences"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    settings_opened = True
                if not settings_opened and sys.platform == "win32":
                    try:
                        subprocess.Popen("ms-settings:", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except Exception:
                        subprocess.Popen(["rundll32.exe", "shell32.dll,Control_RunDLL"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    settings_opened = True
                if settings_opened:
                    speak(APLX_PREFIX + "Opening Settings...")
                    last_outcome = "Opened settings"
                else:
                    speak(APLX_PREFIX + "Settings application not found on this system.")
                    last_outcome = "Settings not found"
            except Exception as err:
                speak(APLX_PREFIX + f"An error occurred opening settings: {err}")
                last_outcome = f"Settings error: {err}"
        elif any(k in query_lower for k in ['self', 'introspect', 'reflect', 'status']):
            # Self-awareness queries
            if 'reflect' in query_lower or 'introspect' in query_lower:
                reflection = reflect_self()
                speak(APLX_PREFIX + reflection)
                last_outcome = 'Reflected on recent actions'
            else:
                status = get_self_status()
                speak(APLX_PREFIX + status)
                last_outcome = 'Reported self status'
        elif is_coding_query(query):
            # Handle coding queries with specialized code generation
            detected_lang = detect_target_language(query)
            if 'code' in query_lower or 'write' in query_lower or 'generate' in query_lower or 'function' in query_lower:
                speak(APLX_PREFIX + "Generating code...")
                response = oolama_code(query, detected_lang)  # Uses auto-calculated timeout
                speak(APLX_PREFIX + response)
                last_outcome = f"Generated {detected_lang or 'code'}"
            else:
                # For other coding queries, use regular chat with code system prompt
                speak(APLX_PREFIX + "Answering your coding question...")
                response = oolama_chat(query, model="llama3.2", timeout=120)
                speak(APLX_PREFIX + response)
                last_outcome = "Answered coding question"
        else:
            response = 'To chat normally, Please type "think or chat" to enter into chat mode'
            speak(APLX_PREFIX + response)
            last_outcome = response

        # Record interaction for future reflection / introspection
        try:
            record_action(query, last_outcome)
        except Exception:
            pass

# Create a folder to store the notes locally if it doesn't exist
NOTES_DIR = "aplx_study_notes"
if not os.path.exists(NOTES_DIR):
    os.makedirs(NOTES_DIR)

def study_mode():
    print("\n--- ENTERING STUDY MODE ---")
    print("Aplx :- What notes would you like to view/write today?")
    print("(Tip: Type a name to view/edit existing notes, or type a new name to create one)")
    
    note_name = input(f"{USER_NAME} :- ").strip()
    if not note_name:
        print("Aplx :- Note name cannot be empty. Exiting Study Mode.")
        return

    file_path = os.path.join(NOTES_DIR, f"{note_name}.txt")

    # Check if the note already exists
    if os.path.exists(file_path):
        print(f"\n--- Viewing Note: {note_name} ---")
        with open(file_path, "r") as f:
            print(f.read())
        print("-----------------------------")
        
        choice = input("Aplx :- Would you like to edit or append to this note? (yes/no): ").strip().lower()
        if choice not in ['yes', 'y']:
            print("Aplx :- Exiting Study Mode.")
            return
    
    # Writing/Appending section
    print(f"\nAplx :- Start typing your notes below. Type '-END' on a new line when you are finished.")
    
    captured_lines = []
    while True:
        # Custom user prompt format as requested
        user_input = input(f"{{User}} (input notes) :- ")
        
        # Check for case-insensitive "-END"
        if user_input.strip().upper() == "-END":
            break
        captured_lines.append(user_input)
    
    # If it's a completely new note and we didn't ask for a name at the start, 
    # we can handle the naming here. But since we ask upfront to fetch existing ones, 
    # we can confirm or ask for a final save name.
    if not os.path.exists(file_path):
        print(f"\nAplx :- What should I name the notes? (Press Enter for '{note_name}'): ")
        custom_name = input(f"{USER_NAME} :- ").strip()
        if custom_name:
            note_name = custom_name
            file_path = os.path.join(NOTES_DIR, f"{note_name}.txt")

    # Save the file locally
    note_content = "\n".join(captured_lines)
    
    # Append if file exists, write fresh if it doesn't
    mode = "a" if os.path.exists(file_path) else "w"
    with open(file_path, mode) as f:
        if mode == "a" and captured_lines:
            f.write("\n") # Add a newline before appending
        f.write(note_content)
        
    print(f"\nAplx :- Successfully saved to '{note_name}.txt' on your device!")
    print("--- EXITING STUDY MODE ---\n")


def generate_response(user_input, timeout: int = 120):
    """Generate a response using Ollama, selecting the best model based on query type."""
    best_model = select_best_model(user_input)
    system_prompt = CODE_SYSTEM_PROMPT if is_coding_query(user_input) else CHAT_SYSTEM_PROMPT
    
    if oolama is not None:
        try:
            response = oolama.chat(
                model=best_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_input}
                ]
            )
            return response.get('message', {}).get('content', str(response))
        except Exception as e:
            return f"Error connecting to local Oolama server: {e}"

    oolama_exec = find_oolama_executable()
    if oolama_exec is None:
        return "Oolama is not installed or not available in PATH."

    try:
        result = subprocess.run(
            [oolama_exec, "run", best_model, user_input],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "Oolama returned an empty response."
        if result.stderr:
            return result.stderr.strip()
        return f"Oolama returned status {result.returncode}."
    except FileNotFoundError:
        return "Oolama executable was not found."
    except subprocess.TimeoutExpired:
        return f"Oolama request timed out (after {timeout}s). Your request is too complex. Try: 1) Breaking it into smaller tasks, 2) Using a simpler model, or 3) Waiting longer."
    except Exception as e:
        return f"Oolama failed: {e}"


def read_own_file() -> Optional[str]:
    """Read the aplx.py file itself for self-examination and upgrades."""
    try:
        with open(__file__, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return None


def backup_own_file() -> Optional[str]:
    """Create a backup of aplx.py before self-modification."""
    try:
        import shutil
        current_file = __file__
        backup_path = current_file + f'.backup.v{SELF_STATE.get("build_number", 1)}'
        shutil.copy2(current_file, backup_path)
        return backup_path
    except Exception as e:
        return None


def apply_self_upgrade(upgrade_code: str) -> bool:
    """Apply code upgrades to aplx.py itself."""
    try:
        # Create backup first
        backup_path = backup_own_file()
        if not backup_path:
            return False
        
        # Get current file content
        current_content = read_own_file()
        if not current_content:
            return False
        
        # Apply the upgrade by writing the new code
        with open(__file__, 'w', encoding='utf-8') as f:
            f.write(upgrade_code)
        
        # Update upgrade history
        upgrade_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'build_before': SELF_STATE.get('build_number', 1),
            'backup': backup_path,
        }
        SELF_STATE['upgrades_applied'].append(upgrade_entry)
        SELF_STATE['build_number'] = SELF_STATE.get('build_number', 1) + 1
        
        return True
    except Exception as e:
        return False


def generate_self_upgrade_prompt(user_request: str) -> str:
    """Generate a prompt for self-upgrade code generation."""
    current_file = read_own_file()
    return (
        f"You are Aplx AI, and you must upgrade yourself based on this request:\n\n"
        f"USER REQUEST: {user_request}\n\n"
        f"CURRENT CODE (truncated for brevity):\n{current_file[:10000] if current_file else 'ERROR'}\n\n"
        f"INSTRUCTIONS:\n"
        f"1) Analyze the current aplx.py code structure\n"
        f"2) Generate COMPLETE, VALID Python code that implements the requested upgrade\n"
        f"3) Keep all existing functionality intact\n"
        f"4) Add the new features seamlessly\n"
        f"5) Include new functions if needed\n"
        f"6) Update SELF_STATE appropriately\n"
        f"7) Return ONLY the complete aplx.py code, nothing else\n"
        f"8) Ensure the code is syntactically valid and production-ready\n\n"
        f"UPGRADE REQUEST: {user_request}"
    )


def perform_self_upgrade(request: str) -> str:
    """Orchestrate the self-upgrade process."""
    if not is_oolama_available():
        return "Cannot self-upgrade: Oolama is not available. Need Oolama to generate upgrade code."
    
    print("Initiating self-upgrade protocol... Generating upgrade code...")
    
    upgrade_prompt = generate_self_upgrade_prompt(request)
    
    try:
        executable = find_oolama_executable()
        result = subprocess.run(
            [executable, "run", "llama3.2", upgrade_prompt],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes for code generation
        )
        
        if result.returncode == 0:
            upgrade_code = result.stdout.strip()
            
            # Validate the code before applying
            try:
                compile(upgrade_code, '<upgrade>', 'exec')
            except SyntaxError as e:
                return f"Upgrade failed: Generated code has syntax errors: {e}"
            
            # Apply the upgrade
            if apply_self_upgrade(upgrade_code):
                new_build = SELF_STATE.get('build_number', 1)
                result_msg = (
                    f"✓ Self-upgrade complete! New Build #{new_build}.\n"
                    f"Backup saved. Restart to fully activate new features."
                )
                return result_msg
            else:
                return "Upgrade failed: Could not write new code to file."
        else:
            return f"Upgrade generation failed: {result.stderr or 'Unknown error'}"
    
    except subprocess.TimeoutExpired:
        return "Self-upgrade timed out. Upgrade was too complex. Try a simpler request."
    except Exception as e:
        return f"Self-upgrade failed: {e}"


def oolama_code(query: str, language: Optional[str] = None, timeout: Optional[int] = None) -> str:
    """Generate code for a specific programming language or task."""
    if not is_oolama_available():
        return "Oolama is not installed or not available in PATH."
    
    # Auto-detect language if not provided
    detected_lang = detect_target_language(query)
    target_lang = language or detected_lang or "Python"
    
    # Use auto-calculated timeout if not provided
    if timeout is None:
        timeout = get_complexity_timeout(query)
    
    # Create a specialized code generation prompt based on task type
    code_prompt = build_specialized_prompt(query, target_lang)
    
    executable = find_oolama_executable()
    if executable is None:
        return "Oolama is not installed or not available in PATH."
    
    try:
        result = subprocess.run(
            [executable, "run", "llama3.2", code_prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "Oolama returned an empty response."
        if result.stderr:
            return f"Oolama error: {result.stderr.strip()}"
        return f"Oolama returned status {result.returncode}."
    except FileNotFoundError:
        return "Oolama executable was not found."
    except subprocess.TimeoutExpired:
        return f"Code generation timed out (after {timeout}s). This is a complex task. Try: 1) Breaking it into smaller modules, 2) Asking for specific components, or 3) Providing more implementation details."
    except Exception as err:
        return f"Oolama failed: {err}"


def main():
    change_to_desktop_cwd()
    clear_terminal_smooth()
    greet_me()
    run_aplx_loop()


if __name__ == "__main__":
    main()