"""kestrel-audio: makes the sound of a Kestrel detection audible and, only when proven safe, cleaner."""
import os

__version__ = os.environ.get("KESTREL_AUDIO_VERSION", "0.0.0-dev")
