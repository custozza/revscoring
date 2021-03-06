from .features import Dictionary, RegexMatches, Stopwords

name = "hungarian"

try:
    import enchant
    dictionary = enchant.Dict("hu")
except enchant.errors.DictNotFoundError:
    raise ImportError("No enchant-compatible dictionary found for 'hu'.  " +
                      "Consider installing 'aspell-hu'.")

dictionary = Dictionary(name + ".dictionary", dictionary.check)
"""
:class:`~revscoring.languages.features.Dictionary` features via
:class:`enchant.Dict` "hungarian".  Provided by `aspell-hu`
"""

stopwords = [
    r"adott",
    r"ahol",
    r"aki",
    r"akik",
    r"akkor",
    r"alap",
    r"alapján",
    r"alatt",
    r"alá",
    r"amely",
    r"ami",
    r"amikor",
    r"amit",
    r"annak",
    r"azonban",
    r"azt",
    r"ban",
    r"ben",
    r"bár",
    r"csak",
    r"egy",
    r"egyes",
    r"egyik",
    r"egyéb",
    r"egyért",
    r"együtt",
    r"egész",
    r"ekkor",
    r"elején",
    r"első",
    r"elő",
    r"először",
    r"előtt",
    r"ennek",
    r"ezek",
    r"ezen",
    r"ezt",
    r"ezzel",
    r"ezért",
    r"kell",
    r"fel",
    r"feletti",
    r"hanem",
    r"hogy",
    r"három",
    r"igen",
    r"illetve",
    r"itt",
    r"kis",
    r"később",
    r"két",
    r"körül",
    r"következő",
    r"között",
    r"közül",
    r"külső",
    r"lehet",
    r"lett",
    r"majd",
    r"meg",
    r"mellett",
    r"mely",
    r"melynek",
    r"mer",
    r"mert",
    r"miatt",
    r"minden",
    r"mint",
    r"mivel",
    r"már",
    r"más",
    r"másik",
    r"második",
    r"még",
    r"nagy",
    r"nagyobb",
    r"nak",
    r"nek",
    r"nem",
    r"nincs",
    r"néhány",
    r"olyan",
    r"pedig",
    r"része",
    r"saját",
    r"sem",
    r"sok",
    r"során",
    r"szerepel",
    r"szerint",
    r"szám",
    r"száma",
    r"számos",
    r"számára",
    r"teljes",
    r"the",
    r"tól",
    r"több",
    r"től",
    r"után",
    r"vagy",
    r"valamint",
    r"való",
    r"van",
    r"volt",
    r"voltak",
    r"vált",
    r"végén",
    r"áll",
    r"állt",
    r"által",
    r"óta",
    r"össze",
    r"úgy",
]

stopwords = Stopwords(name + ".stopwords", stopwords)
"""
:class:`~revscoring.languages.features.Stopwords` features copied from
"common words" in https://meta.wikimedia.org/wiki/?oldid=15534848
"""

badword_regexes = [
    r"anyad",
    r"anyád",
    r"anyádat",
    r"anyátok",
    r"anyátokat",
    r"apád",
    r"asd",
    r"balfasz",
    r"baszni",
    r"baszott",
    r"bazd",
    r"bazdmeg",
    r"bazmeg",
    r"béna",
    r"birkanépet",
    r"birkanépünk",
    r"büdös",
    r"buktája",
    r"buzi",
    r"buzik",
    r"csicska",
    r"csá",
    r"fasszopó",
    r"fasz",
    r"fasza",
    r"faszfej",
    r"faszkalap",
    r"faszok",
    r"faszom",
    r"faszomat",
    r"faszság",
    r"faszt",
    r"faszát",
    r"fing",
    r"fos",
    r"fuck",
    r"geci",
    r"gecik",
    r"gecis",
    r"gecit",
    r"hulye",
    r"hülye",
    r"hülyék",
    r"kabbe",
    r"kaka",
    r"kaki",
    r"kibaszott",
    r"kocsog",
    r"kuki",
    r"kurva",
    r"kurvák",
    r"kurvára",
    r"kurvát",
    r"köcsög",
    r"köcsögök",
    r"lófasz",
    r"megbaszta",
    r"mocskos",
    r"málejku",
    r"mizu",
    r"naon",
    r"picsa",
    r"picsája",
    r"pina",
    r"punci",
    r"putri",
    r"pöcs",
    r"retkes",
    r"ribanc",
    r"rohadt",
    r"sissitek",
    r"szar",
    r"szarok",
    r"szaros",
    r"szart",
    r"szopd",
    r"sále",
    r"elmenyekvolgye",
    r"immoviva",
    r"infosarok",
    r"kirandulastervezo",
    r"kirándulástervező",
    r"magyarvendeglatas",
    r"magyarvendéglátás",
    r"magyarvirtus",
    r"matraonline",
    r"mátraonline",
    r"nosztalgiautazasok",
    r"pestmost",
    r"tapioregio",
    r"turist",
    r"utazasi",
    r"vandorhorgasz",
    r"vándorhorgász",
    r"ellopásával",
    r"eszünkbe",
    r"felelőtlen",
    r"gergényi",
    r"gyurcsány",
    r"hatalmakat",
    r"hazaáruló",
    r"hazudnak",
    r"hazugság",
    r"hazugságra",
    r"hazánknak",
    r"honfitársaim",
    r"hunyadiné",
    r"kicsinyes",
    r"kluboldala",
    r"laci",
    r"lejárató",
    r"lenéznek",
    r"lopásra",
    r"megalázni",
    r"megdézsmálásának",
    r"megvetnek",
    r"megválasztó",
    r"megérdemel",
    r"nemzsidókra",
    r"panamai",
    r"rovástáblás",
    r"szavazófülke",
    r"szex",
    r"talmudjukban",
    r"tiszaeszlár",
    r"toaffot",
    r"tehetetlenül",
    r"torolják",
    r"érdekeik",
    r"érdekeiknek",
]

badwords = RegexMatches(name + ".badwords", badword_regexes)
"""
:class:`~revscoring.languages.features.RegexMatches` features via a list of
badword detecting regexes.
"""

informal_regexes = [
    r"baromság",
    r"dencey",
    r"haha",
    r"hahaha",
    r"hehe",
    r"hello",
    r"hihi",
    r"hülyeség",
    r"képviselőink",
    r"képviselőinket",
    r"képünkbe",
    r"lol",
    r"megválasszuk",
    r"mészárosaim",
    r"országunk",
    r"special",
    r"soknevű",
    r"szavazatunkat",
    r"szeretem",
    r"szeretlek",
    r"szerintem",
    r"szia",
    r"sziasztok",
    r"tex",
    r"xdd",
    r"xddd",
    r"tudjátok",
    r"tönkretesszük",
    r"ugye",
    r"unokáink",
    r"user",
    r"utálom",
    r"vagyok",
    r"vagytok",
]

informals = RegexMatches(name + ".informals", informal_regexes)
"""
:class:`~revscoring.languages.features.RegexMatches` features via a list of
informal word detecting regexes.
"""
