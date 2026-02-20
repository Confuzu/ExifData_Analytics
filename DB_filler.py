"""
DB_filler.py

This script extracts EXIF metadata from images and populates a SQLite database.
It's part of the ExifData Analytics project for analyzing AI-generated image metadata.

Usage: python DB_filler.py
"""

import os
import subprocess
import sys
import platform
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import configparser
from pathlib import Path
import parameter_statistic_DB as db_module

# Check Python version
if sys.version_info < (3, 7):
    print("This script requires Python 3.7 or higher.")
    sys.exit(1)

# Get the directory of the current script
script_dir = Path(__file__).resolve().parent

# Load configuration
config = configparser.ConfigParser()
config_path = script_dir / 'config.ini'
try:
    config.read(config_path)
except configparser.Error as e:
    print(f"Error reading config file: {e}")
    sys.exit(1)

if not config_path.exists():
    print(f"Config file not found: {config_path}")
    sys.exit(1)

# Validate required config sections and options
required_sections = ['Paths', 'Processing', 'Database', 'Security', 'ErrorHandling']
required_options = {
    'Paths': ['logging_dir'],
    'Processing': ['batch_size', 'max_workers'],
    'Database': ['db_name'],
    'Security': ['enable_blocklist', 'custom_blocked_paths', 'allow_network_paths_without_confirmation'],
    'ErrorHandling': ['continue_on_error', 'exiftool_timeout', 'max_file_size_mb']
}

for section in required_sections:
    if section not in config:
        print(f"Missing required section in config: {section}")
        sys.exit(1)
    for option in required_options[section]:
        if option not in config[section]:
            print(f"Missing required option in config: [{section}] {option}")
            sys.exit(1)


# Setup logging
logging_dir = script_dir / config['Paths']['logging_dir']
logging_dir.mkdir(exist_ok=True)

logger = logging.getLogger('DB_filler')
logger.setLevel(logging.DEBUG)

log_file_path = logging_dir / "DB_filler_LOG.txt"
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Constants
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
BATCH_SIZE = int(config['Processing']['batch_size'])
MAX_WORKERS = int(config['Processing']['max_workers'])

# Security settings
ENABLE_BLOCKLIST = config['Security'].getboolean('enable_blocklist')
CUSTOM_BLOCKED_PATHS = config['Security']['custom_blocked_paths']
ALLOW_NETWORK_NO_CONFIRM = config['Security'].getboolean('allow_network_paths_without_confirmation')

# Error handling settings
CONTINUE_ON_ERROR = config['ErrorHandling'].getboolean('continue_on_error')
EXIFTOOL_TIMEOUT = int(config['ErrorHandling']['exiftool_timeout'])
MAX_FILE_SIZE_MB = int(config['ErrorHandling']['max_file_size_mb'])

# ============================================================================
# SECURITY: Defense in Depth - Blocklist + Whitelist
# ============================================================================

def get_blocked_paths() -> List[Path]:
    """
    Return list of paths that should NEVER be scanned.

    This protects against:
    - User accidentally typing /etc or other system paths
    - Malware running as user trying to exfiltrate data
    - Automation bugs in scripts calling this tool

    Reads from config.ini [Security] section for custom additions.

    Returns:
        List of resolved Path objects that are blocked
    """
    # Check if blocklist is disabled (NOT RECOMMENDED)
    if not ENABLE_BLOCKLIST:
        logger.warning("⚠️  SECURITY: Blocklist is DISABLED in config.ini")
        logger.warning("   This is NOT recommended and reduces security!")
        return []

    blocked = [
        # Unix/Linux system directories
        Path('/etc'),
        Path('/root'),
        Path('/sys'),
        Path('/proc'),
        Path('/dev'),
        Path('/boot'),
        Path('/var/log'),
        Path('/var/run'),
        Path('/usr/bin'),
        Path('/usr/sbin'),
        Path('/bin'),
        Path('/sbin'),
        Path('/lib'),
        Path('/lib64'),

        # User sensitive directories
        Path.home() / '.ssh',
        Path.home() / '.gnupg',
        Path.home() / '.aws',
        Path.home() / '.config',
        Path.home() / '.mozilla',
        Path.home() / '.password-store',
        Path.home() / '.local' / 'share' / 'keyrings',
    ]

    # Windows system directories
    if platform.system() == 'Windows':
        blocked.extend([
            Path('C:/Windows'),
            Path('C:/Program Files'),
            Path('C:/Program Files (x86)'),
            Path('C:/ProgramData'),
            Path('C:/System Volume Information'),
        ])

        # Windows user sensitive directories
        appdata = os.environ.get('APPDATA')
        if appdata:
            blocked.append(Path(appdata))

        localappdata = os.environ.get('LOCALAPPDATA')
        if localappdata:
            blocked.append(Path(localappdata))

    # Add custom blocked paths from config
    if CUSTOM_BLOCKED_PATHS.strip():
        custom_paths = [p.strip() for p in CUSTOM_BLOCKED_PATHS.split(',')]
        for custom_path in custom_paths:
            if custom_path:
                blocked.append(Path(custom_path))
                logger.info(f"Added custom blocked path from config: {custom_path}")

    # Resolve all paths and filter out non-existent ones
    resolved_blocked = []
    for p in blocked:
        try:
            if p.exists():
                resolved_blocked.append(p.resolve())
        except (OSError, RuntimeError):
            # Path cannot be resolved, skip it
            continue

    return resolved_blocked


def is_blocked_path(target_path: Path) -> Tuple[bool, str]:
    """
    Check if path is blocked or under a blocked directory.

    Args:
        target_path: Path to check

    Returns:
        (is_blocked, reason) - tuple of bool and explanation string
    """
    try:
        target_resolved = target_path.resolve()
    except (OSError, RuntimeError) as e:
        return True, f"Cannot resolve path: {e}"

    for blocked in get_blocked_paths():
        try:
            # Check if target is the blocked path or a subdirectory of it
            target_resolved.relative_to(blocked)

            # If we get here, target is under blocked path
            return True, f"Path is under blocked directory: {blocked}"

        except ValueError:
            # Not under this blocked path, try next
            continue

    return False, ""


def is_network_path(path: Path) -> bool:
    """
    Check if path is a network path (UNC path on Windows).

    Args:
        path: Path to check

    Returns:
        True if path appears to be a network path
    """
    path_str = str(path)

    # Windows UNC paths: \\server\share or //server/share
    if path_str.startswith('\\\\') or path_str.startswith('//'):
        return True

    # Check for network drive on Windows
    if platform.system() == 'Windows':
        try:
            # Check if path is on a network drive
            result = subprocess.run(
                ['net', 'use'],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Check if the drive letter appears in net use output
            drive_letter = str(path)[0:2]  # e.g., "Z:"
            if drive_letter.upper() in result.stdout:
                return True
        except:
            # If we can't determine, err on the side of caution
            pass

    return False


def validate_file_path(file_path: Path, allowed_root: Path) -> bool:
    """
    Validate that a file is within the allowed directory (WHITELIST).

    This prevents path traversal and symlink escapes AFTER
    the user's chosen directory has passed the blocklist check.

    Args:
        file_path: Path to validate
        allowed_root: The whitelist root directory

    Returns:
        True if path is safe and within allowed_root, False otherwise
    """
    try:
        # Resolve to absolute path (follows symlinks)
        resolved_path = file_path.resolve()

        # Check if resolved path is under allowed_root
        # This catches both path traversal and symlink escapes
        resolved_path.relative_to(allowed_root)

        return True

    except ValueError:
        # relative_to() raises ValueError if path is not under allowed_root
        logger.warning(f"File outside allowed directory: {file_path}")
        logger.warning(f"  Resolved to: {resolved_path}")
        logger.warning(f"  Allowed root: {allowed_root}")
        logger.warning(f"  Skipping file for security.")
        return False

    except (OSError, RuntimeError) as e:
        logger.warning(f"Cannot resolve file path {file_path}: {e}")
        return False


def check_exiftool() -> Tuple[Optional[str], Optional[str]]:
    """Checks if ExifTool is installed and returns its command."""
    try:
        exiftool_cmd = "exiftool.exe" if sys.platform.startswith('win32') else "exiftool"
        subprocess.run([exiftool_cmd, "-ver"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return exiftool_cmd, None
    except FileNotFoundError:
        return None, "ExifTool not found. Please install it to use this script. https://exiftool.org/install.html"
    except subprocess.CalledProcessError:
        return None, "Error occurred while checking ExifTool."

def validate_directory(source_directory: str, include_subfolders: bool = False) -> Tuple[Optional[Path], Optional[str]]:
    """
    Validate directory with defense-in-depth security.

    Security layers:
    1. BLOCKLIST - Prevent scanning dangerous paths (even if user requests)
    2. Path resolution - Follow symlinks, make absolute
    3. Existence check - Verify path exists and is directory
    4. Permission check - Verify readable
    5. Network path warning - Warn about network paths
    6. Returns allowed_root for WHITELIST validation during file scanning

    Args:
        source_directory: User-provided directory path
        include_subfolders: Whether subdirectories will be scanned

    Returns:
        (validated_path, error_message) - Path object or None with error string
    """

    # Step 1: Resolve path (follow symlinks, make absolute)
    try:
        user_path = Path(source_directory).expanduser().resolve()
    except (OSError, RuntimeError) as e:
        error_msg = f"Invalid path '{source_directory}': {e}"
        logger.error(error_msg)
        return None, error_msg

    # Step 2: BLOCKLIST CHECK - First line of defense
    is_blocked, reason = is_blocked_path(user_path)
    if is_blocked:
        logger.error(f"SECURITY: Blocked path access attempt: {user_path}")
        logger.error(f"  Reason: {reason}")

        error_msg = (
            f"❌ Security Error: Cannot scan this directory\n"
            f"   Path: {user_path}\n"
            f"   Reason: {reason}\n\n"
            f"This tool cannot scan system directories or sensitive user paths.\n"
            f"Please choose a directory containing your image files only."
        )
        print(error_msg)
        return None, "Blocked path"

    # Step 3: Verify path exists
    if not user_path.exists():
        error_msg = f"Directory does not exist: {user_path}"
        logger.error(error_msg)
        return None, error_msg

    # Step 4: Verify it's a directory
    if not user_path.is_dir():
        error_msg = f"Path is not a directory: {user_path}"
        logger.error(error_msg)
        return None, error_msg

    # Step 5: Verify readable
    try:
        # Try to list directory to check read permission
        next(user_path.iterdir(), None)
    except PermissionError:
        error_msg = f"No read permission for: {user_path}"
        logger.error(error_msg)
        return None, f"Permission denied to access '{user_path}'."
    except StopIteration:
        # Empty directory is fine
        pass

    # Step 6: Warn if network path
    if is_network_path(user_path):
        logger.warning(f"Network path detected: {user_path}")

        if not ALLOW_NETWORK_NO_CONFIRM:
            print(f"\n⚠️  Warning: Network path detected: {user_path}")
            print()
            print("Scanning network paths can be slow and may access untrusted data.")
            print("Ensure you trust the network source before continuing.")
            print()

            confirm = input("Continue anyway? (yes/no): ").strip().lower()
            if confirm != 'yes':
                print("Cancelled by user.")
                logger.info("User cancelled due to network path warning")
                return None, "User cancelled"
        else:
            logger.info("Network path allowed without confirmation (config setting)")

    # Step 7: Warn if path resolution changed location (symlink was followed)
    original_path = Path(source_directory).expanduser().absolute()
    if original_path != user_path:
        logger.warning(f"Path resolved via symlink:")
        logger.warning(f"  Input:    {source_directory}")
        logger.warning(f"  Resolved: {user_path}")

        print(f"\n⚠️  Notice: Path resolved via symlink")
        print(f"   You entered: {source_directory}")
        print(f"   Resolved to: {user_path}")
        print()

        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Cancelled by user.")
            logger.info("User cancelled due to symlink resolution")
            return None, "User cancelled"

    logger.info(f"✅ Validated directory: {user_path}")
    logger.info(f"   Recursive mode: {include_subfolders}")

    # Return the validated path (becomes allowed_root for whitelist)
    return user_path, None


def collect_image_files(directory: Path, include_subfolders: bool) -> Tuple[List[str], int]:
    """
    Collect image files with whitelist security validation.

    Args:
        directory: The allowed root directory (whitelist)
        include_subfolders: Whether to scan recursively

    Returns:
        (file_list, skipped_count) - List of validated file paths and count of skipped files
    """
    file_list = []
    skipped_count = 0
    allowed_root = directory

    if include_subfolders:
        # Recursive scan
        pattern = directory.rglob('*')
    else:
        # Single directory only
        pattern = directory.glob('*')

    for file_path in pattern:
        # Skip if not a file
        if not file_path.is_file():
            continue

        # Check extension
        if file_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        # SECURITY: Whitelist validation - ensure file is within allowed directory
        if not validate_file_path(file_path, allowed_root):
            print(f"⚠️  Skipped (outside allowed directory): {file_path}")
            skipped_count += 1
            continue

        file_list.append(str(file_path))

    return file_list, skipped_count

def fetch_metadata(filepath: str, exiftool_cmd: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Fetch metadata for a single image file using ExifTool.

    Returns:
        (filepath, metadata, error_message)
        - error_message format: "ERROR_TYPE: details" for categorization
    """
    file_path = Path(filepath)

    # Validation 1: Check file exists
    if not file_path.exists():
        return filepath, None, "FILE_NOT_FOUND: File does not exist"

    # Validation 2: Check file is readable
    try:
        if not os.access(filepath, os.R_OK):
            return filepath, None, "PERMISSION_ERROR: File is not readable"
    except Exception as e:
        return filepath, None, f"PERMISSION_ERROR: {e}"

    # Validation 3: Check file size (avoid hanging on huge files)
    if MAX_FILE_SIZE_MB > 0:
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                return filepath, None, f"FILE_TOO_LARGE: {file_size_mb:.1f}MB (max: {MAX_FILE_SIZE_MB}MB)"
        except Exception as e:
            return filepath, None, f"FILE_SIZE_CHECK_FAILED: {e}"

    # Validation 4: Basic file header check (magic bytes)
    try:
        with open(filepath, 'rb') as f:
            header = f.read(12)
            # Check for common image formats
            is_jpeg = header.startswith(b'\xff\xd8\xff')
            is_png = header.startswith(b'\x89PNG\r\n\x1a\n')
            is_gif = header.startswith(b'GIF')

            if not (is_jpeg or is_png or is_gif):
                return filepath, None, "INVALID_FORMAT: File does not appear to be a valid image"
    except Exception as e:
        return filepath, None, f"FILE_READ_ERROR: {e}"

    # Execute ExifTool with timeout
    try:
        result = subprocess.run(
            [exiftool_cmd, filepath],
            capture_output=True,
            text=True,
            check=True,
            timeout=EXIFTOOL_TIMEOUT
        )
        return filepath, result.stdout, None

    except subprocess.TimeoutExpired:
        return filepath, None, f"EXIFTOOL_TIMEOUT: Exceeded {EXIFTOOL_TIMEOUT}s timeout"

    except subprocess.CalledProcessError as e:
        # ExifTool failed - possibly corrupted file
        stderr_msg = e.stderr.strip() if e.stderr else "Unknown error"
        return filepath, None, f"EXIFTOOL_ERROR: {stderr_msg}"

    except FileNotFoundError:
        return filepath, None, "EXIFTOOL_NOT_FOUND: ExifTool command not found"

    except Exception as e:
        return filepath, None, f"UNEXPECTED_ERROR: {e}"

def update_database_with_images(image_files: List[str], exiftool_cmd: str) -> None:
    """
    Updates the database with metadata from the specified image files.

    Includes robust error handling and statistics tracking.
    """

    db_module.create_table()  # Ensure the table exists before updating the database

    # Track error statistics
    error_stats = {}
    error_details = []
    success_count = 0
    total_count = len(image_files)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(fetch_metadata, filepath, exiftool_cmd) for filepath in image_files]

        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images", unit="file"):
            filepath, metadata, error = future.result()

            if error:
                # Categorize error
                error_type = error.split(':')[0] if ':' in error else "UNKNOWN_ERROR"
                error_stats[error_type] = error_stats.get(error_type, 0) + 1
                error_details.append((filepath, error))
                logger.error(f"Error processing file {filepath}: {error}")

                # Check if we should stop on error
                if not CONTINUE_ON_ERROR:
                    logger.error("CONTINUE_ON_ERROR is False, stopping processing")
                    print(f"\n❌ Error encountered and continue_on_error=false in config")
                    print(f"   Failed file: {Path(filepath).name}")
                    print(f"   Error: {error}")
                    print(f"\nProcessing stopped. {success_count}/{total_count} files processed successfully.")
                    raise RuntimeError(f"Processing stopped due to error: {error}")
            else:
                success_count += 1
                results.append((Path(filepath).name, filepath, metadata))

            # Batch insert into the database
            if len(results) >= BATCH_SIZE:
                db_module.bulk_update_or_insert_metadata(results)
                results.clear()

        # Insert any remaining results
        if results:
            db_module.bulk_update_or_insert_metadata(results)

    # Print summary
    print(f"\n{'='*70}")
    print("PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total files: {total_count}")
    print(f"✅ Successfully processed: {success_count} ({success_count/total_count*100:.1f}%)")
    print(f"❌ Failed: {len(error_details)} ({len(error_details)/total_count*100:.1f}%)")

    if error_stats:
        print(f"\nError Breakdown:")
        for error_type, count in sorted(error_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count}")

        if error_details:
            print(f"\nFirst 5 errors (see log for complete list):")
            for filepath, error in error_details[:5]:
                print(f"  - {Path(filepath).name}: {error}")

    print(f"{'='*70}\n")

    logger.info(f"Finished processing. Success: {success_count}/{total_count}, Errors: {len(error_details)}")
    if error_stats:
        logger.info(f"Error breakdown: {error_stats}")

def main() -> None:
    """Main function to run the DB_filler script."""

    exiftool_cmd, error_message = check_exiftool()
    if error_message:
        logger.error(error_message)
        print(error_message)
        sys.exit(1)

    source_directory = input("Enter the directory to update the database with: ").strip()
    include_subfolders = input("Include subfolders? (yes/no): ").strip().lower() == 'yes'

    # Validate directory with defense-in-depth security
    validated_path, error_message = validate_directory(source_directory, include_subfolders)
    if error_message:
        if error_message not in ["User cancelled", "Blocked path"]:
            logger.error(error_message)
            print(error_message)
        sys.exit(1)

    # Collect files with whitelist validation
    logger.info("Collecting image files...")
    image_files, skipped_count = collect_image_files(validated_path, include_subfolders)

    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} files outside allowed directory (symlink escapes)")
        print(f"\n⚠️  Skipped {skipped_count} files for security (outside allowed directory)")

    if not image_files:
        logger.warning("No valid image files found in the specified directory.")
        print("No valid image files found in the specified directory.")
        sys.exit(0)

    logger.info(f"Found {len(image_files)} valid image files to process")
    print(f"\n✅ Found {len(image_files)} valid image files")

    update_database_with_images(image_files, exiftool_cmd)
    print("Database update completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.exception("An unexpected error occurred:")
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
