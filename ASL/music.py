import gtts
import playsound

text="A"
sound= gtts.gTTS(text,lang="en")
sound.save("sign.mp3")
playsound.playsound("C:/Users/aswin/OneDrive/Desktop/ASL/sign.mp3")
