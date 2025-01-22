import gtts
import playsound

text="A"
sound= gtts.gTTS(text,lang="en")
sound.save("sign.mp3")
playsound.playsound("C:\Users\aswin\Desktop\Sign-Language-text-to-audio-conversion\ASL\sign.mp3")
