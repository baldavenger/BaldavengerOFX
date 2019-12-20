SequenceParsing
===============

Definition:
----------

Pattern: A file name, absolute or not, which is composed of symbols (variable) which
can take several values.

#Supported variables:

- %d : indicates that any number should be present instead of the variable. This option
works like printf like arguments, e.g:  %04d, which indicates you want the variable
to match only numbers with 4 prepending zeroes.

- \#\#\# : Same as %d but the number of "#" character indicates how many digits the number
should be.

- %v: indicates that a short view name should be present instead of the variable. 
This means either "l" for the left view and "r" for the right view.

- %V: Same as %v but for long view names, i.e: "right" and "left"


This repository aims to help solving 3 problems:

1) Given a file name, search within the same directory for all other files that matches
the same "pattern".

2) Given a "pattern" generate file names for a given frame number and view.

3) Given a files list, tries to group files under similar patterns.

