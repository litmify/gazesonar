#!/usr/bin/env python3
"""
Test script for frame-based TTS functionality
"""
import sys
import time
import pyttsx3
import settings

def test_tts():
    """Test TTS engine and settings"""
    print("Testing Text-to-Speech functionality...")
    print(f"TTS Mode: {settings.TTS_MODE}")
    print(f"Announce every {settings.TTS_ANNOUNCE_EVERY_N_FRAMES} frames")
    print(f"Announce all objects: {settings.TTS_ANNOUNCE_ALL_OBJECTS}")

    # Initialize TTS
    engine = pyttsx3.init()
    engine.setProperty('rate', settings.TTS_VOICE_RATE)
    engine.setProperty('volume', settings.TTS_VOICE_VOLUME)

    # Test announcements
    test_messages = [
        "Detected person",
        "Detected 2 chairs and 1 laptop",
        "Detected: person, chair, laptop, cell phone"
    ]

    for i, message in enumerate(test_messages, 1):
        print(f"\nTest {i}: {message}")
        engine.say(message)
        engine.runAndWait()
        time.sleep(1)

    print("\nTTS test completed successfully!")

if __name__ == "__main__":
    test_tts()