#!/usr/bin/env python3

import pyttsx3
import time

def test_tts():
    print("Testing Text-to-Speech functionality...")

    engine = pyttsx3.init()

    # Set properties
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

    # Get available voices
    voices = engine.getProperty('voices')
    print(f"Available voices: {len(voices)}")

    # Test phrases for different objects
    test_phrases = [
        "Person",
        "Laptop",
        "Chair",
        "Book",
        "Phone"
    ]

    print("\nTesting object announcements:")
    for phrase in test_phrases:
        print(f"Announcing: {phrase}")
        engine.say(phrase)
        engine.runAndWait()
        time.sleep(1)

    print("\nTTS test completed successfully!")

if __name__ == "__main__":
    test_tts()