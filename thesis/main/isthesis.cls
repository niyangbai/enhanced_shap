%%%%%%%%%%%%%%

% Nothing to do here!

%%%%%%%%%%%%%%


\NeedsTeXFormat{LaTeX2e}

\ProvidesClass{isthesis}

% Declare key=value options
\RequirePackage{kvoptions}[2010/12/23] % Important: Require version 3.10
% Earlier versions of this package had a bug concerning the handling of \DeclareDefaultOption{…}

% Conditional commands
\RequirePackage{ifthen}

% Namespacing
\SetupKeyvalOptions{
	family=isthesis,
	prefix=isthesis@
}
%% Set English as default language
\DeclareStringOption[]{language}[english]

% Set Master Thesis as default document type
\DeclareStringOption[]{type}[master]

% Process options
\ProcessKeyvalOptions*

% Inherit from report class
\LoadClass[12pt]{report}

% Use UTF-8 for input encoding
\RequirePackage[utf8]{inputenc}

% Use T1 font encoding
\RequirePackage[T1]{fontenc}

% Key value processing
\RequirePackage{pgfkeys}

% Graphics
\RequirePackage{graphicx}
\RequirePackage{adjustbox}

% Colors
\RequirePackage{xcolor}

% Mathematics
\RequirePackage{amsmath}
\RequirePackage{amssymb}

% Algorithms
\RequirePackage{algorithm}
\RequirePackage[noend]{algpseudocode}
\RequirePackage{algorithmicx}

% Use Times New Roman as default font
\RequirePackage{helvet}
\renewcommand{\familydefault}{\sfdefault}

% Use Garamond as default font
%\RequirePackage[urw-garamond]{mathdesign}

% Floats
\RequirePackage{float}
\RequirePackage{placeins} 

% Heading formatting
\RequirePackage{fancyhdr}

\RequirePackage{titlesec}

% Managing table of contents appearances
\RequirePackage{tocvsec2}

% Change the resetting of counters
\RequirePackage{chngcntr}

% Advanced tables
\RequirePackage{array}
\RequirePackage{tabularx}
\RequirePackage{multirow}
\RequirePackage{tablefootnote}
\RequirePackage{booktabs}

% External tables
\RequirePackage{pgfplotstable}
\pgfplotsset{compat=1.12}

% Advanced captions
\RequirePackage{caption}
\RequirePackage{subcaption}

% Code listings
\RequirePackage{listings}

% Minipages with variable lists
\RequirePackage{varwidth}

% Advanced table of contents formatting
\RequirePackage[titles]{tocloft}

% Formatting of ordered (enumeration) and unordered (itemize) lists
\RequirePackage[inline]{enumitem}

% Customizing of citations in the text
\RequirePackage[
    style=apa,
    backend=biber,
    natbib=true]
    {biblatex}

% Advanced page layout
\RequirePackage{geometry}

% Hyperlinks and PDF information
\RequirePackage[hidelinks]{hyperref}

% Glossaries (must be loaded after hyperref)
\RequirePackage[acronym,nonumberlist,nopostdot,notree,nomain]{glossaries}

% Length calculations
\RequirePackage{calc}

% Allow for "Textmarker" highlighting \texthl{highlighted text}
\RequirePackage{soul}

% \todo Notes
\setlength{\marginparwidth}{2cm}
\RequirePackage{todonotes}

% Abbreviations
%\RequirePackage[acronym]{glossaries}

% Blind texts
\RequirePackage{lipsum}

 % For correct spacing after abbrevations
 \RequirePackage{xspace}
 \DeclareRobustCommand\xdot{.\@\xspace}
 
% FAU colors
\definecolor{fau-blue}{RGB}{0,74,159}
\definecolor{fau-orange}{RGB}{253,183,53}
\definecolor{fau-red}{RGB}{197,15,60}
\definecolor{fau-cyan}{RGB}{24,180,241}
\definecolor{fau-green}{RGB}{123,183,37}
\definecolor{fau-grey}{RGB}{140,159,177}

 
 % Processing of advanced options
 % (keys are can be defined later)
 \newcommand{\isthesis}[1]{
 	\setkeys{isthesis}{#1}%
 }
 
 % Macro for creating options
 \newcommand{\isthesis@new@option}{\@ifstar{\isthesis@new@option@star}{\isthesis@new@option@nostar}}
 
 % Non-starred version (name, internal name, default value)
 \newcommand{\isthesis@new@option@nostar}[3]{%
 	\isthesis@new@option@star{#1}{#2}{#3}{\expandafter\renewcommand\csname isthesis@#2\endcsname{##1}}%
 }
 
 % Starred version (name, internal name, default value, macro to execute)
 \newcommand{\isthesis@new@option@star}[4]{%
 	\expandafter\newcommand\csname isthesis@#2\endcsname{#3}%
 	\define@key{isthesis}{#1}{#4}%
 }
 
 % %
 % % Localization
 % %
 \ifthenelse{\equal{\isthesis@language}{english}}{
	 % ENGLISH
	 
	 % English hyphenation rules
	 \RequirePackage[ngerman, english]{babel}
	 
	  % Use nice quotation marks with \enquote
	 \RequirePackage{csquotes}
	 
	% Date formatting
	\RequirePackage[en-GB,showdow]{datetime2}

	\DTMnewdatestyle{isthesisdate}{%
		\renewcommand*{\DTMdisplaydate}[4]{\DTMenglishordinal{##3} \DTMenglishmonthname{##2}  ##1}%
		\renewcommand*{\DTMDisplaydate}{\DTMdisplaydate}%
	}
	\isthesis@new@option{label-principal}{label@principal}{Principal Supervisor}
	\isthesis@new@option{label-associate}{label@associate}{Associate Supervisor}
	\isthesis@new@option{label-tutor}{label@tutor}{Tutor}
	\isthesis@new@option{label-author}{label@author}{Presented by}
	\isthesis@new@option{label-submission}{label@submission}{Submission}
	\isthesis@new@option{label-atthe}{label@atthe}{at the}
	\isthesis@new@option{label-atthex}{label@atthex}{at}
	\isthesis@new@option{label-context}{label@context}{in the context of the seminar}
	\isthesis@new@option{secondary-logo}{secondary@logo}{assets/institut.pdf}
	\newcommand{\listabbreviationname}{Abbreviations}
	\newcommand{\listsymbolname}{Symbols}
	\newcommand{\authorshipstatementname}{Declaration of Academic Integrity}
	\newcommand{\consentformname}{Consent Form}%
	\AtBeginDocument{%
		\renewcommand{\contentsname}{Contents}%
		\renewcommand{\listfigurename}{Figures}%
		\renewcommand{\lstlistlistingname}{Listings}%
		\renewcommand{\listtablename}{Tables}%
		\renewcommand{\abstractname}{Abstract} %
		\renewcommand{\appendixname}{Appendices} %
		\renewcommand{\bibname}{References}%
		\setlength{\cftfignumwidth}{2cm}
		\renewcommand{\chaptermark}[1]{\markboth{\MakeUppercase{#1}}{}}
	}
}{\ifthenelse{\equal{\isthesis@language}{german}}{
	% GERMAN
	
	 % English hyphenation rules
	 \RequirePackage[ngerman, english]{babel}
	 
	 % Use nice quotation marks with \enquote
	 \RequirePackage[style=ngerman]{csquotes}
	 
  	% Date formatting
  	\RequirePackage[german, showdow]{datetime2}

  	\DTMnewdatestyle{isthesisdate}{%
  		\renewcommand*{\DTMdisplaydate}[4]{\DTMgermanordinal{##3}. \DTMgermanmonthname{##2}  ##1}%
  		\renewcommand*{\DTMDisplaydate}{\DTMdisplaydate}%
  	}
	 % Allows the use of "," as a decimal point
	 \RequirePackage{icomma}
	 
	\isthesis@new@option{label-principal}{label@principal}{Themenersteller}
	\isthesis@new@option{label-associate}{label@associate}{Betreuer}
	\isthesis@new@option{label-tutor}{label@tutor}{Tutor}
	\isthesis@new@option{label-author}{label@author}{vorgelegt von}
	\isthesis@new@option{label-submission}{label@submission}{Abgabetermin}
	\isthesis@new@option{label-atthe}{label@atthe}{am}
	\isthesis@new@option{label-atthex}{label@atthex}{an der}
	\isthesis@new@option{label-context}{label@context}{im Rahmen des Seminars}
	\isthesis@new@option{secondary-logo}{secondary@logo}{assets/institut.pdf}
	\newcommand{\listsymbolname}{Symbolverzeichnis}
	\newcommand{\listabbreviationname}{Abk\"urzungsverzeichnis}
	\newcommand{\authorshipstatementname}{Abschlie{\ss}ende Erkl\"arung}
	\newcommand{\consentformname}{Einverst\"andniserkl\"arung}%
	\AtBeginDocument{%
		\renewcommand{\contentsname}{Inhaltsverzeichnis}%
		\renewcommand{\listfigurename}{Abbildungsverzeichnis}%
		\renewcommand{\lstlistlistingname}{Codeverzeichnis}%
		\renewcommand{\listtablename}{Tabellenverzeichnis}%
		\renewcommand{\abstractname}{Abstract} %
		\renewcommand{\appendixname}{Anhang} %
		\renewcommand{\bibname}{Literaturverzeichnis}%
		\renewcommand{\figurename}{Abbildung}
		\renewcommand{\cftfigpresnum}{Abbildung\ }
		\setlength{\cftfignumwidth}{2.3cm}
		\renewcommand{\tablename}{Tabelle}
		\renewcommand{\cfttabpresnum}{Tabelle\ }
		\renewcommand{\ALG@name}{Algorithmus} %algorithms package
		\renewcommand{\chaptermark}[1]{\markboth{\MakeUppercase{#1}}{}}
	}
}{
%  \ClassError{isthesis}{Unknown language ``\isthesis@language'' specified}{}
}}

% Set specific settings w.r.t. thesis type
 \ifthenelse{\equal{\isthesis@type}{master}}{
 	% Specific settings for Master Theses
 }{\ifthenelse{\equal{\isthesis@type}{bachelor}}{
 	% Specific settings for Bachelor Theses
 }{\ifthenelse{\equal{\isthesis@type}{seminar}}{
 	% Specific settings for Seminar Theses
 }{
 \ClassError{isthesis}{Unknown type ``\isthesis@type'' specified}{}
}}}
 
% Macro for creating a thesis type
% (name, language, title page caption)
\newcommand{\isthesis@new@type}[3]{%
	\providecommand{\isthesis@title@type}{}%
	\ifthenelse{\equal{\isthesis@type}{#1}}{
		\ifthenelse{\equal{\isthesis@language}{#2}}{
			\renewcommand{\isthesis@title@type}{#3}%
		}{\relax}
	}{\relax}
}

% Create thesis types
\isthesis@new@type{seminar}{german}{Seminararbeit}
\isthesis@new@type{seminar}{english}{Term Paper}
\isthesis@new@type{bachelor}{german}{Bachelorarbeit}
\isthesis@new@type{bachelor}{english}{Bachelor Thesis}
\isthesis@new@type{master}{english}{Master Thesis}
\isthesis@new@type{master}{german}{Masterarbeit}

% Helper to remove line breaks from a command
\newcommand{\removelinebreaks}[1]{%
  \begingroup\def\\{ }#1\endgroup%
}

% Regular font style for hyperlinks
\urlstyle{same}

% Set default page layout
\geometry{paper=a4paper,left=3.5cm, top=2.5cm, bottom=2.5cm, right=1.5cm, twoside}

% Disable indention of paragraphs
\setlength{\parindent}{0pt}

% Paragraph spacing
\newlength{\parskip@default}
\setlength{\parskip@default}{\parskip}
\AtBeginDocument{%
    \setlength{\parskip}{12pt}
}

% Line spacing
\newlength{\baselineskip@default}
\setlength{\baselineskip@default}{\baselineskip}
\AtBeginDocument{%
    \setlength{\baselineskip}{18pt}%
}

% Page style (override default plain style)
\fancypagestyle{plain}{
	\fancyhf{}
	\fancyhead[LO,RE]{\nouppercase{\leftmark}}
	\fancyhead[RO,LE]{\thepage}
	\setlength{\headheight}{20pt} 
	\renewcommand{\headrulewidth}{1pt} %set to 0 for no headrule
}

% Footnote formatting
\RequirePackage[flushmargin,hang,bottom]{footmisc}
\setlength{\footnotemargin}{0.75cm}
\renewcommand\footnoterule{\kern-3pt \hrule width 2in \kern 2.6pt} % Length of separator between text and footnotes
\interfootnotelinepenalty=10000 % Prevent footnotes to be split over multiple pages

% Sectioning formatting
\titleformat{\chapter}{\normalfont\large\bfseries}{\makebox[1.5cm][l]{\thechapter}}{0pt}{}
\titlespacing*{\chapter}{0pt}{-16pt plus 2pt minus 2pt}{8pt plus 2pt minus 2pt}

\titleformat{\section}{\normalfont\bfseries}{\makebox[1.5cm][l]{\thesection}}{0pt}{}
\titlespacing*{\section}{0pt}{12pt plus 2pt minus 2pt}{6pt plus 2pt minus 2pt}

\titleformat{\subsection}{\normalfont\bfseries}{\makebox[1.5cm][l]{\thesubsection}}{0pt}{}
\titlespacing{\subsection}{0pt}{12pt plus 2pt minus 2pt}{6pt plus 2pt minus 2pt}

% Numberless sectioning formatting (sub headings)
\titleformat{name=\section,numberless}{\normalfont\itshape}{}{0pt}{}
\titlespacing{name=\section,numberless}{0pt}{4pt plus 2pt minus 2pt}{4pt plus 2pt minus 2pt}

\titleformat{name=\subsection,numberless}{\normalfont\itshape}{}{0pt}{}
\titlespacing{name=\subsection,numberless}{0pt}{4pt plus 2pt minus 2pt}{4pt plus 2pt minus 2pt}

\titleformat{name=\subsubsection,numberless}{\normalfont\itshape}{}{0pt}{}
\titlespacing{name=\subsubsection,numberless}{0pt}{4pt plus 2pt minus 2pt}{4pt plus 2pt minus 2pt}

% Keep floats within the sections they are defined in
\newcommand{\isthesis@section@barrier}[1]{%
    \expandafter\let\csname #1@default\expandafter\endcsname\csname #1\endcsname%
    % \expandafter\renewcommand\csname #1\endcsname{\FloatBarrier\expandafter\FloatBarrier\csname #1@default\endcsname}%
    \expandafter\renewcommand\csname #1\endcsname{\FloatBarrier\csname #1@default\endcsname}%
    % using l@addto@macro
    % or \noexpand primitive?
}

\isthesis@section@barrier{chapter}
\isthesis@section@barrier{section}
\isthesis@section@barrier{subsection}

% Caption formatting
\captionsetup{figurewithin=none,tablewithin=none} % Continuous numbering
\captionsetup{labelsep=quad} % No separator between label and caption
\AtBeginDocument{
	    \counterwithout{lstlisting}{chapter} % Continous numbering for listings
}

% Figure formatting
\let\figure@default\figure
\let\endfigure@default\endfigure

\pgfkeys{
    /isthesis/figure/.cd,
    caption/.initial={},
    toc/.initial={},
}

\define@key{figure}{caption}{\newcommand{\figure@caption}{#1}}
\define@key{figure}{toc}{\newcommand{\figure@toc}{#1}}
\define@key{figure}{label}{\newcommand{\figure@label}{#1}}
\define@key{figure}{placement}{\newcommand{\figure@placement}{#1}}
\define@key{figure}{width}{\newcommand{\figure@width}{#1}}
\define@key{figure}{fit}[]{\newcommand{\figure@fit}{}}
\define@key{figure}{cache}{\newcommand{\figure@cache}{#1}}
\define@key{figure}{cachename}{\newcommand{\figure@cachename}{#1}}

\newcommand{\figure@private@resize@width}{}
\newcommand{\figure@private@resize@proxy}[1]{\resizebox{\figure@private@resize@width}{!}{#1}}

\renewenvironment{figure}[1][]{%
    \setkeys{figure}{#1}%
    \ifthenelse{\isundefined{\figure@placement}}{%
        \newcommand{\figure@internal@placement}{ht}%
    }{%
        \ifthenelse{\equal{\figure@placement}{exact}}{% 
            \newcommand{\figure@internal@placement}{!ht}%{H}%
        }{%
            \ifthenelse{\equal{\figure@placement}{top}}{% 
                \newcommand{\figure@internal@placement}{t}%
            }{%
                \ifthenelse{\equal{\figure@placement}{bottom}}{% 
                    \newcommand{\figure@internal@placement}{b}%
                }{}%
            }%
        }%
    }%
    \expandafter\figure@default\expandafter[\figure@internal@placement]%
    \ifthenelse{\isundefined{\figure@cache}}{}{%
        % Control caching
        \tikzset{external/export next=\figure@cache}%
        % Raise a warning if a figure is not cached
        % \ifthenelse{\equal{\figure@cache}{false}}{%
        %     \ClassWarning{isthesis}{Figure \figure@cachename is not cached.}%
        % }%
    }%
    \ifthenelse{\isundefined{\figure@cachename}}{}{%
        % Set cache name
        \tikzsetnextfilename{\figure@cachename}%
    }%
    \center%
    \ifthenelse{\isundefined{\figure@width}}{}{%
        \renewcommand{\figure@private@resize@width}{\figure@width}%
        \collect@body\figure@private@resize@proxy%
    }%
}{%
    \ifthenelse{\isundefined{\figure@toc}}{%
        % Without dedicated TOC caption
        \caption{\figure@caption\label{\figure@label}}%
    }{%
        % With dedicated TOC caption
        \caption[\figure@toc]{\figure@caption\label{\figure@label}}%
    }%
    \endcenter%
    \endfigure@default%
}
\renewcommand{\captionlabelfont}{\bfseries}

% Table formatting
\let\table@default\table
\let\endtable@default\endtable

\define@key{table}{caption}{\newcommand{\table@caption}{#1}}
\define@key{table}{toc}{\newcommand{\table@toc}{#1}}
\define@key{table}{label}{\newcommand{\table@label}{#1}}
\define@key{table}{columns}{\newcommand{\table@columns}{#1}}
\define@key{table}{placement}{\newcommand{\table@placement}{#1}}

\renewenvironment{table}[1][]{%
    \begingroup%
    \newcommand{\table@internal@placement}{ht}%
    \setkeys{table}{#1}%
    \expandafter\table@default\expandafter[\table@internal@placement]%
    % \table@default[ht]
    % \footnotesize%
    \center%
    \renewcommand{\arraystretch}{1.15}%
    % \tabular{\table@columns}
}{%
    % \endtabular%
    \ifthenelse{\isundefined{\table@toc}}{%
        % Without dedicated TOC caption
        \caption{\table@caption\label{\table@label}}%
    }{%
        % With dedicated TOC caption
        \caption[\table@toc]{\table@caption\label{\table@label}}%
    }%
    \endcenter%
    \endtable@default%
    \endgroup%
}

% Itemization formatting
\setlist[itemize]{
%    label={$-$}, %activate for dashes instead of bullets
	labelindent=24pt,
	labelsep=18pt,
	leftmargin=*,
	topsep=0pt,
}

% Enumeration formatting
\setlist[enumerate, 1]{
    label={(\arabic*)},
    align=left,
	labelindent=24pt,
	labelsep=18pt,
	leftmargin=*,
	topsep=0pt,
}

% Display verbatim environments
\let\verbatim@default\verbatim
\let\endverbatim@default\endverbatim

\pgfkeysdef{/isthesis/verbatim/font}{#1}

\newcommand{\isthesis@verbatim@inner}[1][]{%
    {\nfss@catcodes\scantokens{\gdef\isthesis@verbatim@parameters{#1}}}%
    \isthesis@verbatim@parameters%
    % \pgfkeys{/isthesis/verbatim/.cd,\isthesis@verbatim@parameters}%
}

\renewenvironment{verbatim}{%
    \bgroup%
    \small%
    \setlength{\topsep}{0pt}%
    \setlength{\partopsep}{0pt}%
    \verbatim@default%
    \isthesis@verbatim@inner%
}{%
    \endverbatim@default%
    \egroup%
}

\clubpenalty=10000
\widowpenalty=10000
\displaywidowpenalty=10000

% Continue footnote numbering across chapters
\counterwithout{footnote}{chapter}

% \ Listings Set up
\lstset{ %
	basicstyle=\ttfamily\footnotesize,
	numbers=left,                   % where to put the line-numbers
	numberstyle=\footnotesize,      % the size of the fonts that are used for the line-numbers
	stepnumber=0,                   % Each X line will be numbered
	numbersep=5pt,                  % how far the line-numbers are from the code
	backgroundcolor=\color{white},  % choose the background color. 
	commentstyle=\color{fau-blue},
	keywordstyle=\color{fau-red},
	showspaces=false,               % show spaces adding particular underscores
	showstringspaces=false,         % underline spaces within strings
	showtabs=false,                 % show tabs within strings adding particular underscores
	%frame=single,           % adds a frame around the code
	tabsize=2,          % sets default tabsize to 2 spaces
	captionpos=b,           % sets the caption-position to bottom
	breaklines=true,        % sets automatic line breaking
	breakatwhitespace=false,    % sets if automatic breaks should only happen at whitespace
	escapeinside={\%*}{*)},       % if you want to add a comment within your code
	moredelim=**[is][\color{fau-orange}]{@}{@},
}

% Title page
\let\maketitle@default\maketitle
\renewcommand{\maketitle}{%
    \begin{titlepage}%
		\pagenumbering{gobble}
    	\DTMsetdatestyle{isthesisdate}
        \newgeometry{top=1cm,outer=2.5cm,bottom=2.5cm,inner=2.5cm}
        \includegraphics[height=\isthesis@primary@logo@height]{\isthesis@primary@logo}%
        \hfill%
        \ifthenelse{\equal{\isthesis@secondary@logo}{}}{}{\includegraphics[height=\isthesis@secondary@logo@height]{\isthesis@secondary@logo}}%
        \vfill%
        \begin{center}%
        %    {\large \isthesis@author@name~\isthesis@author@lastname \par}%
         %   \vskip 1.5em%
            {\bfseries \Large \@title \par}%
            {\large \isthesis@sub@title \par}
            \vskip 3em%
            {\bfseries \large \isthesis@title@type \par}%
            \ifthenelse{\equal{\isthesis@type}{seminar}}{ \isthesis@label@context \- \enquote{\isthesis@seminar} \par}{}%
            \ifthenelse{\equal{\isthesis@group@name}{}}{}{%
                \vskip 3em%
                {%
                    \ifthenelse{\equal{\isthesis@group@institute}{}}{}{ 
                        \isthesis@label@atthex ~\isthesis@group@university \\
                        \isthesis@label@atthe ~\isthesis@group@institute}\\
                    \textbf{\isthesis@group@name}%
                    \par%
                }%
            }
            \ifthenelse{\equal{\isthesis@associate@group@name}{}}{}{%
                \vskip 1.5em%
                {%
                    in collaboration with the \isthesis@associate@group@name%
                    \ifthenelse{\equal{\isthesis@associate@group@institute}{}}{}{\\ (\isthesis@associate@group@institute)}%
                    \par%
                }%
            }
        \end{center}\par%
        \vfill%
        \begin{center}%
            \parbox{0cm}{%
                \begin{tabbing}%
				\= \hspace*{5cm} \= \kill
				\> \isthesis@label@principal: \> \isthesis@supervisor@principal \\ %
				\> \isthesis@label@associate: \> \isthesis@supervisor@associate \\%
				\ifthenelse{\equal{\isthesis@supervisor@tutor}{}}{\relax}{ %
				\> \isthesis@label@tutor: \> \isthesis@supervisor@tutor \\} %
				\> \> \\
				\> \isthesis@label@author: \> \isthesis@author@name \\ %
				\ifthenelse{\equal{\isthesis@author@address}{}}{\relax}{ %
					\> \> \isthesis@author@address \\} %
				\ifthenelse{\equal{\isthesis@author@city }{}}{\relax}{ %
					\> \> \isthesis@author@zip \, \isthesis@author@city  \\}%
				\ifthenelse{\equal{\isthesis@author@phone}{}}{\relax}{ %
					\> \> \isthesis@author@phone \\} %
				\ifthenelse{\equal{\isthesis@author@email}{}}{\relax}{ %
					\> \> \isthesis@author@email \\
				    %\> \> \isthesis@author@studies \\
				} %
				\ifthenelse{\equal{\isthesis@author@matriculation}{}}{\relax}{ %
					\> \> \isthesis@author@matriculation \\} %
				\> \> \\
					\> \isthesis@label@submission: \> {\isthesis@submission@date \DTMusedate{submissiondate}}
                \end{tabbing}%
            }%
        \end{center}\par%
        \restoregeometry%
        \renewcommand{\headrulewidth}{0pt}
        \cleardoublepage
		\pagenumbering{Roman}
    \end{titlepage}%
} 

% Customizable logos on title page
\isthesis@new@option{primary-logo}{primary@logo}{assets/FB_Logo.png}
\isthesis@new@option{primary-logo-height}{primary@logo@height}{14mm}
% Secondary logo is language dependent and defined above
\isthesis@new@option{secondary-logo-height}{secondary@logo@height}{16mm}

% Override latex default meta information macros
\let\default@title\title
\renewcommand{\title}[1]{%
    \default@title{#1}%
    \hypersetup{pdftitle={#1}}%
}

\let\default@author\author
\renewcommand{\author}[1]{%
    \default@author{#1}%
    \hypersetup{pdfauthor={#1}}%
}

% Additional meta information variables
\isthesis@new@option*{title}{title}{Thesis Title}{\AtBeginDocument{\title{#1}}}
\isthesis@new@option{sub-title}{sub@title}{Thesis Subtitle}
\isthesis@new@option{author-name}{author@name}{Firstname}
\isthesis@new@option{author-email}{author@email}{student@fau.de}
\isthesis@new@option{author-address}{author@address}{F\"urther Stra{\ss}e" 248}
\isthesis@new@option{author-phone}{author@phone}{+49 xxxxxxxxx}
\isthesis@new@option{author-matriculation}{author@matriculation}{123456}
\isthesis@new@option{author-zip}{author@zip}{90429}
\isthesis@new@option{author-city}{author@city}{N\"urnberg"}
\isthesis@new@option{studies}{author@studies}{International Information Systems}
\isthesis@new@option{group}{group@name}{Digital Industrial Service Systems}
\isthesis@new@option{group-institute}{group@institute}{}
\isthesis@new@option{group-university}{group@university}{Friedrich-Alexander-Universität Erlangen-Nürnberg}
\isthesis@new@option{associate-group}{associate@group@name}{}
\isthesis@new@option{associate-group-institute}{associate@group@institute}{}
\isthesis@new@option{principal-supervisor}{supervisor@principal}{} %Themenersteller
\isthesis@new@option{associate-supervisor}{supervisor@associate}{}
\isthesis@new@option{tutor-supervisor}{supervisor@tutor}{} %Betreuer
\isthesis@new@option{seminar}{seminar}{}
\isthesis@new@option*{submission-date}{submission@date}{}{\DTMsavedate{submissiondate}{#1}}
% Roman page numbering from the beginning
\AtBeginDocument{%
    \pagenumbering{Roman}%
	\author{\isthesis@author@name}%
}

% Start with II after title page
\let\maketitle@default\maketitle
\renewcommand{\maketitle}{%
    \maketitle@default%
    \setcounter{page}{1}%
}


\renewenvironment{abstract}
{
    \clearpage
    \phantomsection
    \markboth{\abstractname}{\abstractname}
    % \addcontentsline{toc}{chapter}{\abstractname}
    
    \begin{center}
        \large\textbf{Abstract}
    \end{center}
    
    \hrule
    \small
    \list{}{\leftmargin=1.5cm \rightmargin=1.5cm}
    \item\relax
}
{
    \endlist
    \vspace{10pt}
    \hrule
    \pagebreak
    
}


% Content wrapper
\newenvironment{content}{%
	% specific spacing in content
	\titlespacing*{\chapter}{0pt}{-30pt plus 2pt minus 2pt}{8pt plus 2pt minus 2pt}
    \clearpage
    % Enable arabic page numbering
    \pagenumbering{arabic}
    }
    {
    %% Post content
    \pagebreak
    }

% Quote page
\pgfkeys{
    /isthesis/quotepage/.cd,
    author/.initial={},
    citation/.initial={}
}

\newcommand{\quotepage}[2][]{
    \clearpage%
    \begingroup%
    \pgfkeys{/isthesis/quotepage/.cd,#1}%
    \thispagestyle{empty}%
    \null\vfill%
    \centering%
    \begin{varwidth}{\textwidth}
        {\raggedright\large\itshape%
        #2\par\bigskip%
        }%
        {\pgfkeys{/isthesis/quotepage/author/.get=\author}%
        \raggedleft\large\scshape\author\par%
        }
    \end{varwidth}%
    \pgfkeys{/isthesis/quotepage/citation/.get=\citationkey}%
    \nocite{\citationkey}%
    \vfill\vfill%
    \endgroup%
    \clearpage%
}

% Table of contents
\let\default@tableofcontents\tableofcontents
\renewcommand{\tableofcontents}{%
    % \addcontentsline{toc}{chapter}{\contentsname}%
    {%
        \clearpage%
        \phantomsection%
     
        \pdfbookmark[0]{\contentsname}{contents}%
        \setlength{\parskip}{0pt}%
        % Set depth of table of contents to subsection
        \setcounter{tocdepth}{2}%
        \default@tableofcontents%
        \markboth{\contentsname}{\contentsname}
    }%
    \pagebreak%
}

\renewcommand{\cftdotsep}{2}
\setlength{\cftparskip}{0pt}

\renewcommand{\cftchapfont}{}
\renewcommand{\cftchappagefont}{}
\renewcommand{\cftchapleader}{\cftdotfill{\cftdotsep}}
\setlength{\cftbeforechapskip}{0pt}

% List of figures
\let\default@listoffigures\listoffigures
\renewcommand{\listoffigures}{%
    \phantomsection%
    \addcontentsline{toc}{chapter}{\listfigurename}%
    {%
        \setlength{\parskip}{0pt}%
        \default@listoffigures%
    }%
    \markboth{\listfigurename}{\listfigurename}
    \pagebreak%
}

\renewcommand{\cftfigindent}{0pt}


% List of tables
\let\default@listoftables\listoftables
\renewcommand{\listoftables}{%
    \phantomsection%
    \addcontentsline{toc}{chapter}{\listtablename}%
    {%
        \setlength{\parskip}{0pt}%
        \default@listoftables%
    }%
    \markboth{\listtablename}{\listtablename} % For the header to not be capitalised
    \pagebreak%
}

\renewcommand{\cfttabindent}{0pt}
\setlength{\cfttabnumwidth}{2cm}


% List of listings
\let\default@lstlistoflistings\lstlistoflistings
\renewcommand{\lstlistoflistings}{%
    \phantomsection%
    \addcontentsline{toc}{chapter}{\lstlistlistingname}%
    {%
        \setlength{\parskip}{0pt}%
        \default@lstlistoflistings%
    }%
    \markboth{\lstlistlistingname}{\lstlistlistingname}
    \pagebreak%
}
\renewcommand\l@lstlisting[2]{\@dottedtocline{1}{0pt}{2em}{\lstlistingname~#1}{#2}}


% Formatting of glossaries (using tabbing environment)
\newglossarystyle{isthesis}{%
    % Redefine the glossary environment
    \renewenvironment{theglossary}{\begingroup\setlength{\baselineskip}{\baselineskip@default}}{\endgroup}%
    \renewenvironment{theglossary}{}{}%
    % No space at the beginning of the glossary
    \renewcommand*{\glossaryheader}{}%
    % No space before and after group headings
    \renewcommand*{\glsgroupheading}[1]{}%
    \renewcommand*{\glsgroupskip}{}%
    % Formatting of entries
    \renewcommand*{\glossaryentryfield}[5]{%
        \begingroup%
        \setlength{\parskip}{0pt}%
        \setlength{\hangindent}{\cfttabnumwidth}%
        \par%
        \parbox[t]{\cfttabnumwidth}{\glstarget{##1}{##2}}%
        ##3%
        \endgroup%
    }%
    % Formatting of sub entries
    \renewcommand*{\glossarysubentryfield}[6]{%
        \glossaryentryfield{##2}{##3}{##4}{##5}{##6}%
    }%
}


% List of abbreviations
\newcommand{\listofabbreviations}{%
    \phantomsection%
    \addcontentsline{toc}{chapter}{\listabbreviationname}%
    {%
        \setlength{\parskip}{0pt}%
        \printglossary[type=\acronymtype,title={\listabbreviationname}, style=isthesis]%
    }%
    \pagebreak%
}
% Symbol type
\newcommand{\symboltype}{glssymbol}
\newglossary{\symboltype}{gls}{glo}{\listsymbolname}
\newcounter{symbol}
\setcounter{symbol}{0}
\newcommand{\addsymboltolist}[3]{\addtocounter{symbol}{1}\newglossaryentry{sym\thesymbol}{name={#1},description={#3},sort={#2},type=\symboltype}}

\AtBeginDocument{
	\renewcommand*{\@gls@adjustmode}{}
	\glsaddall[types={\symboltype}]
}

% List of symbols
\newcommand{\listofsymbols}{%
    \phantomsection%
    \addcontentsline{toc}{chapter}{\listsymbolname}%
    {%
        \setlength{\parskip}{0pt}%
        \printglossary[type=\symboltype,title={\listsymbolname},style=isthesis]%
    }%
    \pagebreak
}

% Appendix
\let\appendix@default\appendix
\let\appendix\undefined
\newenvironment{appendix}{%
    \appendix@default%
    \chapter*{\appendixname}\markboth{\appendixname}{\appendixname}%
    \addcontentsline{toc}{chapter}{\appendixname}%
    \renewcommand{\thesection}{\Alph{section}}%
    \settocdepth{chapter}%
}{%
    \resettocdepth%
}

% Format of equation numbering
\renewcommand{\theequation}{\thechapter.\arabic{equation}}


\newcommand{\references}{%
    \addcontentsline{toc}{chapter}{\bibname}%
    \printbibliography[title=\bibname]
    \pagebreak%
}

% Declaration of authorship
\define@key{authorshipstatement}{pagenumbering}[true]{\renewcommand{\authorshipstatement@pagenumbering}{#1}}

\newcommand{\authorshipstatement}[1][]{%
    \begingroup%
    \newcommand{\authorshipstatement@pagenumbering}{true}%

    \setkeys{authorshipstatement}{#1}%
    \clearpage%
    \phantomsection%
    \markboth{\authorshipstatementname}{\authorshipstatementname}
    \pdfbookmark[0]{\authorshipstatementname}{authorshipstatement}%
    \ifthenelse{\equal{\authorshipstatement@pagenumbering}{false}}{%
        % Disable page numbering
        \pagenumbering{gobble}%
    }{}%
    \ifthenelse{\equal{\authorshipstatement@pagenumbering}{only}}{%
        % Render a page without content except for the page numbering
        \mbox{}%
    }{%
        \chapter*{\authorshipstatementname}%
        \DTMsetdatestyle{isthesisdate}
        \ifthenelse{\equal{\isthesis@language}{german}}{
					Hiermit versichere ich, \isthesis@author@name~ \isthesis@author@matriculation, die vorgelegte Arbeit selbstständig und ohne unzulässige Hilfe Dritter sowie ohne die Hinzuziehung nicht offengelegter und insbesondere nicht zugelassener Hilfsmittel angefertigt zu haben. Die Arbeit hat in gleicher oder ähnlicher Form noch keiner anderen Prüfungsbehörde vorgelegen und wurde auch von keiner anderen Prüfungsbehörde bereits als Teil einer Prüfung angenommen.

                    Die Stellen der Arbeit, die anderen Quellen im Wortlaut oder dem Sinn nach entnommen wurden, sind durch Angaben der Herkunft kenntlich gemacht. Dies gilt auch für Zeichnungen, Skizzen, bildliche Darstellungen sowie für Quellen aus dem Internet.

                    Mir ist insbesondere bewusst, dass die Nutzung künstlicher Intelligenz verboten ist, sofern diese nicht ausdrücklich als Hilfsmittel von dem Prüfungsleiter bzw. der Prüfungsleiterin zugelassen wurde. Dies gilt insbesondere für Chatbots (insbesondere ChatGPT) bzw. allgemein solche Programme, die anstelle meiner Person die Aufgabenstellung der Prüfung bzw. Teile derselben bearbeiten könnten.

                    Des Weiteren ist mir bekannt, dass die gemeinsame Bearbeitung der Aufgabenstellung mit anderen Personen in einem Raum oder mithilfe sozialer Medien eine unzulässige Hilfe Dritter im o.g. Sinne darstellt, wenn nicht ausdrücklich Gruppenarbeit vorgesehen ist. Jeder Austausch mit anderen Personen mit Ausnahme von Prüfenden und Aufsichtführenden während der Prüfungszeit über Aufbau oder Inhalte der Prüfung oder Informationen (z.B. Quellen) ist unzulässig. Gleiches gilt für den Versuch der jeweiligen Handlung.

                    Verstöße gegen die o.g. Regeln sind als Täuschung bzw. Täuschungsversuch zu qualifizieren und führen zu einer Bewertung der Prüfung mit „nicht bestanden“. Weiterführende Sanktionen sind bei mehrfachen oder auch besonders drastischen Regelverstößen durch den Prüfungsausschuss möglich.
			
			\isthesis@author@city, den \today \\[2\baselineskip]%
			\isthesis@author@name%
        }{
			         Hiermit versichere ich, \isthesis@author@name~  \isthesis@author@matriculation , die vorgelegte Arbeit selbstständig und ohne unzulässige Hilfe Dritter sowie ohne die Hinzuziehung nicht offengelegter und insbesondere nicht zugelassener Hilfsmittel angefertigt zu haben. Die Arbeit hat in gleicher oder ähnlicher Form noch keiner anderen Prüfungsbehörde vorgelegen und wurde auch von keiner anderen Prüfungsbehörde bereits als Teil einer Prüfung angenommen.

                    Die Stellen der Arbeit, die anderen Quellen im Wortlaut oder dem Sinn nach entnommen wurden, sind durch Angaben der Herkunft kenntlich gemacht. Dies gilt auch für Zeichnungen, Skizzen, bildliche Darstellungen sowie für Quellen aus dem Internet.

                    Mir ist insbesondere bewusst, dass die Nutzung künstlicher Intelligenz verboten ist, sofern diese nicht ausdrücklich als Hilfsmittel von dem Prüfungsleiter bzw. der Prüfungsleiterin zugelassen wurde. Dies gilt insbesondere für Chatbots (insbesondere ChatGPT) bzw. allgemein solche Programme, die anstelle meiner Person die Aufgabenstellung der Prüfung bzw. Teile derselben bearbeiten könnten.

                    Des Weiteren ist mir bekannt, dass die gemeinsame Bearbeitung der Aufgabenstellung mit anderen Personen in einem Raum oder mithilfe sozialer Medien eine unzulässige Hilfe Dritter im o.g. Sinne darstellt, wenn nicht ausdrücklich Gruppenarbeit vorgesehen ist. Jeder Austausch mit anderen Personen mit Ausnahme von Prüfenden und Aufsichtführenden während der Prüfungszeit über Aufbau oder Inhalte der Prüfung oder Informationen (z.B. Quellen) ist unzulässig. Gleiches gilt für den Versuch der jeweiligen Handlung.

                    Verstöße gegen die o.g. Regeln sind als Täuschung bzw. Täuschungsversuch zu qualifizieren und führen zu einer Bewertung der Prüfung mit „nicht bestanden“. Weiterführende Sanktionen sind bei mehrfachen oder auch besonders drastischen Regelverstößen durch den Prüfungsausschuss möglich.

			
			\isthesis@author@city,  \today\\[2\baselineskip]%
			\isthesis@author@name
        }
    }%
    \ifthenelse{\equal{\authorshipstatement@pagenumbering}{false}}{%
        % Restore default page numbering
        \clearpage%
        \pagenumbering{arabic}%
    }{}%
    \endgroup
}
% consent form
\define@key{consentform}{pagenumbering}[true]{\renewcommand{\consentform@pagenumbering}{#1}}
\newcommand{\consentform}[1][]{%
	\begingroup%
	\newcommand{\consentform@pagenumbering}{true}%
	
	\setkeys{consentform}{#1}%
	\clearpage%
	\phantomsection%
	\markboth{\consentformname}{\consentformname}
	\pdfbookmark[0]{\consentformname}{consentform}%
	\ifthenelse{\equal{\consentform@pagenumbering}{false}}{%
		% Disable page numbering
		\pagenumbering{gobble}%
	}{}%
	\ifthenelse{\equal{\consentform@pagenumbering}{only}}{%
		% Render a page without content except for the page numbering
		\mbox{}%
	}{%
	\chapter*{\consentformname}%
	\DTMsetdatestyle{isthesisdate}
	\ifthenelse{\equal{\isthesis@language}{german}}{
	{\small 
	zur Prüfung meiner Arbeit mit einer Software zur Erkennung von Plagiaten
	
	\textbf{Name}:~\isthesis@author@name \\
    \textbf{Matrikelnummer}:~\isthesis@author@matriculation\\
    \textbf{Studiengang}:~\isthesis@author@studies \\
	\textbf{Adresse}:~\isthesis@author@address, \isthesis@author@zip~\isthesis@author@city}  \\
	\textbf{Titel der Arbeit}:~\enquote{\removelinebreaks{\@title}}


	
	
	\textbf{Was ist ein Plagiat?}
	Als ein Plagiat wird eine Übernahme fremden Gedankengutes in die eigene Arbeit angesehen, bei der die Quelle, aus der die Übernahme erfolgt, nicht kenntlich gemacht wird. Es ist dabei unerheblich, ob z.B. fremde Texte wörtlich übernommen werden, nur Strukturen (z.B. argumentative Figuren oder Gliederungen) aus fremden Quellen entlehnt oder Texte aus einer Fremdsprache übersetzt werden.  
	
	\textbf{Softwarebasierte Überprüfung}
	Alle Bachelor- und Masterarbeiten werden vom Prüfungsamt mit Hilfe einer entsprechenden Software auf Plagiate geprüft. Die Arbeit wird zum Zweck der Plagiatsüberprüfung an einen Software-Dienstleister übermittelt und dort auf Übereinstimmung mit anderen Quellen geprüft. Zum Zweck eines zukünftigen Abgleichs mit anderen Arbeiten wird die Arbeit dauerhaft in einer Datenbank gespeichert. Ein Abruf der Arbeit ist ausschließlich durch die Friedrich-Alexander Universität Erlangen-Nürnberg möglich. Der Studierende erklärt sich damit einverstanden, dass allein zum beschriebenen Zweck der Plagiatsprüfung die Arbeit dauerhaft gespeichert und vervielfältigt werden darf. Das Ergebnis der elektronischen Plagiatsprüfung wird dem Erstgutachter mitgeteilt.  
	
	\textbf{Sanktionen}
	Liegt ein Plagiat vor, ist dies ein Täuschungsversuch i.S. der Prüfungsordnung, durch den die Prüfungsleistung als \enquote{nicht bestanden} gewertet wird. Es erfolgt eine Mitteilung an das Prüfungsamt und die dortige Dokumentation. In schwerwiegenden Täuschungsfällen kann der Prüfling von der Prüfung insgesamt ausgeschlossen werden. Dies kann unter Umständen die Exmatrikulation bedeuten. Plagiate können auch nach Abschluss des Prüfungsverfahrens und Verleihung des Hochschulgrades zum Entzug des erworbenen Grades führen.  
	
	
	Hiermit erkläre ich, dass ich die obigen Ausführungen gelesen habe und mit dem Verfahren zur Aufdeckung und Sanktionierung von Plagiaten einverstanden bin.  

	\vspace{1em}
	
	\isthesis@author@city,  \today \\[2\baselineskip]%
	\isthesis@author@name
	}
	{\small 
	for the use of plagiarism detection software to check my thesis
	

	\textbf{Name}:~\isthesis@author@name \\
	\textbf{Student number}:~\isthesis@author@matriculation\\
	\textbf{Course of study}:~\isthesis@author@studies \\
	\textbf{Address}:~\isthesis@author@address, \isthesis@author@zip~\isthesis@author@city}  \\
	\parbox{\textwidth}{\textbf{Title of the thesis}:~\enquote{\removelinebreaks{\@title}}}


	
	\textbf{What is plagiarism?}
	Plagiarism is defined as submitting someone else's work or ideas as your own without a complete indication of the source. It is hereby irrelevant whether the work of others is copied word by word without acknowledgment of the source, text structures (e.g. line of argumentation or outline) are borrowed or texts are translated from a foreign language. 
	
	\textbf{Use of plagiarism detection software}
	The examination office uses plagiarism software to check each submitted bachelor and master thesis for plagiarism. For that purpose the thesis is electronically forwarded to a software service provider where the software checks for potential matches between the submitted work and work from other sources. For future comparisons with other theses, your thesis will be permanently stored in a database. Only the Friedrich-Alexander-University Erlangen-Nürnberg is allowed to access your stored thesis. The student agrees that his or her thesis may be stored and reproduced only for the purpose of plagiarism assessment. The first examiner of the thesis will be advised on the outcome of the plagiarism assessment. 
	
	\textbf{Sanctions}
	Each case of plagiarism constitutes an attempt to deceive in terms of the examination regulations and will lead to the thesis being graded as \enquote{failed}. This will be communicated to the examination office where your case will be documented. In the event of a serious case of deception the examinee can be generally excluded from any further examination. This can lead to the exmatriculation of the student. Even after completion of the examination procedure and graduation from university, plagiarism can result in a withdrawal of the awarded academic degree. 
	
	
	I confirm that I have read and understood the information in this document. I agree to the outlined procedure for plagiarism assessment and potential sanctioning. 

	\vspace{1em}
	
	\isthesis@author@city,  \today \\[2\baselineskip]%
	\isthesis@author@name
}
%
\ifthenelse{\equal{\consentform@pagenumbering}{false}}{%
	% Restore default page numbering
	\clearpage%
	\pagenumbering{arabic}%
}{}%
\endgroup
}

\pagestyle{plain}
\makeindex
%\makeglossaries