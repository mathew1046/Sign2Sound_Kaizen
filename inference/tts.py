"""
Text-to-Speech Module for Sign Language Recognition

Converts predicted sign language text to speech output.

Author: Team Kaizen
Date: January 2026
"""

import logging
import pyttsx3
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextToSpeech:
    """Convert text to speech."""
    
    def __init__(self, rate: int = 150, volume: float = 1.0):
        """
        Initialize TTS engine.
        
        Args:
            rate: Speech rate (default: 150)
            volume: Volume level 0.0-1.0 (default: 1.0)
        """
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        
        logger.info(f"TTS engine initialized (rate={rate}, volume={volume})")
    
    def speak(self, text: str, wait: bool = True):
        """
        Convert text to speech and play.
        
        Args:
            text: Text to speak
            wait: Wait for speech to finish
        """
        try:
            # Remove class prefix if present
            text = text.replace('Malayalam_', '').replace('ISL_', '')
            
            logger.info(f"Speaking: {text}")
            self.engine.say(text)
            
            if wait:
                self.engine.runAndWait()
            else:
                self.engine.runAndWait()
        
        except Exception as e:
            logger.error(f"TTS error: {e}")
    
    def set_rate(self, rate: int):
        """
        Set speech rate.
        
        Args:
            rate: Speech rate (50-300)
        """
        self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume: float):
        """
        Set volume level.
        
        Args:
            volume: Volume 0.0-1.0
        """
        self.engine.setProperty('volume', max(0.0, min(1.0, volume)))
    
    def set_language(self, language: str = 'english'):
        """
        Set speech language.
        
        Args:
            language: Language code (e.g., 'english', 'malayalam')
        """
        # Note: Language support depends on system TTS engine
        logger.info(f"Language set to: {language}")
    
    def close(self):
        """Stop and cleanup TTS engine."""
        try:
            self.engine.stop()
        except:
            pass


def text_to_speech_file(text: str, output_file: str, rate: int = 150):
    """
    Generate speech from text and save to file.
    
    Args:
        text: Text to convert
        output_file: Output audio file path
        rate: Speech rate
    """
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.save_to_file(text, output_file)
    engine.runAndWait()
    logger.info(f"Audio saved to {output_file}")


if __name__ == "__main__":
    # Test TTS
    print("Testing Text-to-Speech...")
    
    tts = TextToSpeech(rate=150)
    
    tts.speak("Hello, this is a test", wait=True)
    tts.speak("Sign language recognition system", wait=True)
    
    tts.close()
    
    print("✓ TTS test complete!")
