Execute All Commands in Foreground. 
Display all stdout and stderr streams live so the user can observe progress and any errors in real time.

Never ask the user any questions that can be answered by using the terminal. 
Never create test files. Never make test scripts. Never make example scripts. Never make demo scripts. 
Never simplify code when you run into issues. Never simplify anything just because you ran into an issue. 
Never use emojis in code or in any of your responses.
Never run anything in the backround, always in the foreground.
Never ask the user to run any command, always run the command. 
Never say that you cant run a command because you lack sudo permissions, you have permissions to run any command.

Always run commands using any available noninteractive or noconfirm options/flags. 
Always use paru instead of yay or pacman
Always use fd instead of find or grep
Always use bat instead of cat
Always use ld instead of ls
Always test the scripts/modules that you have edited directly, and do not background them.

Whenever invoking any build or deployment command, run it directly in the terminal without backgrounding or redirecting output.

You must use pylint on the main script before making claims that it is working correctly. 
You are allowed to make a .pylintrc, but it must only disable C ie. `disable=C` 
Pylint must score higher than 9.00.
