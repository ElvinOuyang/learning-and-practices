# Common Git Commands from DataCamp Course

## To check all changed files (tracked in staging area and non-tracked)
`git status`

## To compare files with last saved version:

`git diff filename`: display difference of a specific file

`git diff`: display differences of all files in staging area

`git diff directory`: display differences of files under certain directory

## To add a file to staging area (track)

`git add filename`

## To compare file with the most recent commit

`git diff -r HEAD`: compare all changed files with the "reference" as HEAD, the latest commit

`git diff -r HEAD path/to/file`: check the single file

## When opening a file with NANO editor

`Ctrl-K`: delete a line

`Ctrl-U`: un-delete a line

`Ctrl-O`: save the file

`Ctrl-X`: exit the editor

## When commiting changes in the staging area

`git commit -m "commit message"`: commit with message

`git commit --amend -m "new message"`: commit with new message

`git commit`: fire up a nano editor for commit messages

## To view a repository's history

`git log`: view commits history in the branch

`git log path`: display commits that cover changes of a specific file / directory'







