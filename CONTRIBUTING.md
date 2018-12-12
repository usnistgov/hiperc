# Contributing to [HiPerC: High Performance Computing Strategies for Boundary Value Problems][_hiperc]

Thank you for spending some time with the HiPerC project.
I sincerely appreciate your interest, and hope you will consider contributing
to the documentation, source code, and community at large.

There are a lot of coprocessor architectures and threading models out there, and
we strive for the highest possible performance to ensure valid benchmarking.
If your pull request meets the guidelines below, we'll be able to assess and
merge your contribution much more quickly. If not, don't worry &mdash; we'll
guide you through a constructive code review.

The phase-field accelerator benchmarks are open-source, and we really appreciate
input from the community. There are several ways you can contribute, including
editing or adding documentation, writing tutorials, submitting bug reports,
requesting new features, and writing code to incorporate. If you'd like to help
out, please do!

If you're looking for support or guidance, please visit our [Gitter room][_gitter]
to talk with the developers and community members, or send an e-mail to
trevor.keller@nist.gov.

## Ground Rules

- Help us to maintain a considerate and supportive community by reading and
  enforcing the [Code of Conduct][_conduct]. Be nice to newcomers.
- Promote clean code by ensuring your code and documentation build with neither
  warnings nor errors using compiler flags equivalent to [GCC's][_gcc]
  ```-Wall -pedantic```.
- Engage with the community by creating [issues][_issue] for changes or
  enhancements you'd like to make. Discuss ideas the open and get feedback.
- Maximize reusability by keeping code simple. Program in C if possible, and
  generally follow the [Google C++ Style Guide][_goog]. Avoid creating new
  classes if possible.
- Document new functions and classes with [Doxygen][_doxy]-compatible comments
  in the source code.

## Getting Started

Interested in helping, but unsure where to start? Consider proofreading the PDF
documentation and reporting or fixing errors! There's bound to be a typo in
there. Or look through the existing [issues][_issue] for fixes or enhancements
you're interested in. The number of comments on an issue can indicate its level
of difficulty, as well as the impact closing it will have.

If you're brand new to open source, welcome! You might benefit from the
[GitHub Help Pages][_ghhelp] and a [tutorial][_tut].

## Pull Requests

We use [git][_git] version control with a [branching workflow][_branch],
summarized below. Please do not commit directly to the ```master``` branch.

1. Create a fork of [HiPerC][_hiperc] on your personal
   GitHub account.
2. For obvious changes, such as typos or edits to ```.gitignore```, you can edit
   your fork directly in the browser, then file a [pull request][_pr].
3. For most changes, clone your fork to your local machine, then create a
   working branch off of ```master```. If you're working, for example, on issue
   #42, *summarize usage in pseudocode*, create a branch called
   ```issue42_summarize-usage-in-pseudocode``` by executing
   ```git checkout -b issue42_summarize-usage-in-pseudocode master```.
4. Make changes on the issue branch. In your commit messages, use the keywords
   ["Addresses" or "Closes"][_ghkey] where appropriate.
5. When finished, push the *working branch* to your fork on GitHub, *e.g.*
   ```git push origin issue42_summarize-usage-in-pseudocode```.
6. Visit GitHub and make the pull request official.

Obvious fixes will be merged quickly. Enhancements may undergo a code review,
which we typically conduct using [reviewable][_review].

## Happy Coding!

[_hiperc]:    https://github.com/usnistgov/hiperc
[_branch]:  http://nvie.com/posts/a-successful-git-branching-model/
[_conduct]: https://github.com/usnistgov/hiperc/blob/master/CODE_OF_CONDUCT.md
[_doxy]:    http://www.doxygen.nl/manual/docblocks.html
[_gcc]:     https://gcc.gnu.org/
[_ghhelp]:  https://help.github.com/
[_ghkey]:   https://help.github.com/articles/closing-issues-using-keywords/
[_git]:     https://git-scm.com/
[_gitter]:  https://gitter.im/usnistgov/hiperc
[_goog]:    https://google.github.io/styleguide/cppguide.html
[_issue]:   https://github.com/usnistgov/hiperc/issues
[_pr]:      https://help.github.com/articles/about-pull-requests/
[_review]:  https://reviewable.io/reviews/usnistgov/hiperc
[_tut]:     https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github
