## About
Per request, here is my mostly working sea-monster loot calculating discord bot. It comes with no license (written or implied) and support will probably be nonexistant. While this program is not intentionally malicious in any way and does not record or store any data (outside of memory), it also was written with no regard for security or privacy.

## Requirements:
* Python 3.6 (or theoretically any version >3 and <3.7) 
* Pytesseract and Tesseract OCR version 3.05.02 (this is hard to find but some German university hosts an installer somewhere)
* Everything else imported at the top of the file (I suggest PIP)
* An existing discord application with a bot. The token should be plug-and-play.

## Usage:
Set up your bot accout through discord, add the token to `monsterbotpublic.py`. To minimize CPU usage from parsing EVERY image uploaded to a discord server, I suggest using permissions to only allow the bot access to a text channel in which you only record sea-monster loot. There is also an empty string in the file which accepts the id of the channel you would like the bot to observe. This string can be obtained by enabling discord developer mode, right-clicking on the channel and clicking `copy-id`.

Once your bot is invited to the server and you have restricted its channel access, it is time to run the bot.

The program acts like a server and runs until force quit. It takes no arguments (nor does it accept any). Once your dependencies are all configured, you should be able to run with the following command:

```py -3.6 monsterbotpublic.py```

If possible, I recommend that you have your guildies crop any screenshots to just outside the boundary of the items for best results (see `sample_1.jpg` in `/example_output/`)

## Debugging:
If you find issues and wish to refine the image recognition, the `monster_test.py` file runs a discord-less version of the program. It loops through the contents of the `images` directory and outputs what it finds with an OpenCV viewer, as well as printing detected values to console. The images in `images` are a mixture of well-functioning input, as well as some examples of input that produces errors which I never got around to fixing.

## Fonts:
If you or someone in your guild uses a custom font, accuracy of the bot may decrease significantly. Tesseract OCR can be trained on custom fonts, but I have not done so since it is a pain.

# \<BRUTAL> WINS AGAIN
\<BRUTAL> WINS AGAIN
