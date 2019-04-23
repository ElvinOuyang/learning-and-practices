# Useful commands using `screen` application in Unix/Ubuntu systems

**Using `screen` to keep process running on the AWS EC2 without hangup**

I recently started to run python scripts on my AWS EC2 instances to gather tweet streams about new games. The always-on nature of a low-power instance (t2.micro) with relatively small SSD volume (20G) can keep my twitter streaming listener running for days (check out the codes I used for tweet streaming [here](https://github.com/ElvinOuyang/social-media-mining)). However, when I was using the EC2 instances with SSH connections, I found out that the instance will shut down the process I initiated through SSH soon after I disconnect. I then did some research online and figured out that you can keep your ssh session running without being disrupted by a "hangup" by putting your sessions in "screens" with the `screen` app.

Once you have the `screen` app ready in your ubuntu system, use the following commands to manage multiple processes that you wish you can have your cloud instance running on the cloud without interruption.

**To list all existing screens**

`$screen -ls`

**To return to a previous screen listed with above command**

`$screen -r screen-id`
`$screen -r screen-title`

**To start a new screen with a specified screen title**

`$screen -S screen-title`

**While in a screen session, use following shortcuts to:**

**CTRL-A C**: create new screen

**CTRL-A N**: next screen

**CTRL-A P**: previous screen

**CTRL-A D**: detach current screen session

The single most useful shortcut is the above **CTRL-A D**, with which you can
detach your session from your ssh interface with it running in the background
on the cloud. You can quit your ssh session without hesitation and come back
with the `$screen -r` commands to resume / check the progress. Just be sure that
you can financially cover the time the instance was running on the cloud while
you were away! (I used the free tier AWS instance for my screen sessions so I
have next to nothing to pay for my screen sessions.)
