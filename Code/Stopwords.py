# This file only contains lists of stop words in different languages.
# We import and use the stop words like this because we found it greatly reduces runtime.

# Custom stop word list (German)
stop_word_list_de = {"a", "ab", "aber", "ach", "acht", "achte", "achten", "achter", "achtes", "ag", "alle", "allein",
                     "allem", "allen", "aller", "allerdings", "alles", "allgemeinen", "als", "also", "am", "an",
                     "ander",
                     "andere", "anderem", "anderen", "anderer", "anderes", "anderm", "andern", "anderr", "anders", "au",
                     "auch", "auf", "aus", "ausser", "ausserdem", "außer", "außerdem", "b", "bald", "bei", "beide",
                     "beiden", "beim", "beispiel", "bekannt", "bereits", "besonders", "besser", "besten", "bin", "bis",
                     "bisher", "bist", "c", "d", "d.h", "da", "dabei", "dadurch", "dafür", "dagegen", "daher", "dahin",
                     "dahinter", "damals", "damit", "danach", "daneben", "dank", "dann", "daran", "darauf", "daraus",
                     "darf", "darfst", "darin", "darum", "darunter", "darüber", "das", "dasein", "daselbst", "dass",
                     "dasselbe", "davon", "davor", "dazu", "dazwischen", "daß", "dein", "deine", "deinem", "deinen",
                     "deiner", "deines", "dem", "dementsprechend", "demgegenüber", "demgemäss", "demgemäß", "demselben",
                     "demzufolge", "den", "denen", "denn", "denselben", "der", "deren", "derer", "derjenige",
                     "derjenigen",
                     "dermassen", "dermaßen", "derselbe", "derselben", "des", "deshalb", "desselben", "dessen",
                     "deswegen",
                     "dich", "die", "diejenige", "diejenigen", "dies", "diese", "dieselbe", "dieselben", "diesem",
                     "diesen", "dieser", "dieses", "dir", "doch", "dort", "drei", "drin", "dritte", "dritten",
                     "dritter",
                     "drittes", "du", "durch", "durchaus", "durfte", "durften", "dürfen", "dürft", "e", "eben",
                     "ebenso",
                     "ehrlich", "ei", "ei,", "eigen", "eigene", "eigenen", "eigener", "eigenes", "ein", "einander",
                     "eine",
                     "einem", "einen", "einer", "eines", "einig", "einige", "einigem", "einigen", "einiger", "einiges",
                     "einmal", "eins", "elf", "en", "ende", "endlich", "entweder", "er", "ernst", "erst", "erste",
                     "ersten", "erster", "erstes", "es", "etwa", "etwas", "euch", "euer", "eure", "eurem", "euren",
                     "eurer", "eures", "f", "folgende", "früher", "fünf", "fünfte", "fünften", "fünfter", "fünftes",
                     "für",
                     "g", "gab", "ganz", "ganze", "ganzen", "ganzer", "ganzes", "gar", "gedurft", "gegen", "gegenüber",
                     "gehabt", "gehen", "geht", "gekannt", "gekonnt", "gemacht", "gemocht", "gemusst", "genug",
                     "gerade",
                     "gern", "gesagt", "geschweige", "gewesen", "gewollt", "geworden", "gibt", "ging", "gleich", "gott",
                     "gross", "grosse", "grossen", "grosser", "grosses", "groß", "große", "großen", "großer", "großes",
                     "gut", "gute", "guter", "gutes", "h", "hab", "habe", "haben", "habt", "hast", "hat", "hatte",
                     "hatten", "hattest", "hattet", "heisst", "her", "heute", "hier", "hin", "hinter", "hoch", "hätte",
                     "hätten", "i", "ich", "ihm", "ihn", "ihnen", "ihr", "ihre", "ihrem", "ihren", "ihrer", "ihres",
                     "im",
                     "immer", "in", "indem", "infolgedessen", "ins", "irgend", "ist", "j", "ja", "jahr", "jahre",
                     "jahren",
                     "je", "jede", "jedem", "jeden", "jeder", "jedermann", "jedermanns", "jedes", "jedoch", "jemand",
                     "jemandem", "jemanden", "jene", "jenem", "jenen", "jener", "jenes", "jetzt", "k", "kam", "kann",
                     "kannst", "kaum", "kein", "keine", "keinem", "keinen", "keiner", "keines", "kleine", "kleinen",
                     "kleiner", "kleines", "kommen", "kommt", "konnte", "konnten", "kurz", "können", "könnt", "könnte",
                     "l", "lang", "lange", "leicht", "leide", "lieber", "los", "m", "machen", "macht", "machte", "mag",
                     "magst", "mahn", "mal", "man", "manche", "manchem", "manchen", "mancher", "manches", "mann",
                     "mehr",
                     "mein", "meine", "meinem", "meinen", "meiner", "meines", "mensch", "menschen", "mich", "mir",
                     "mit",
                     "mittel", "mochte", "mochten", "morgen", "muss", "musst", "musste", "mussten", "muß", "mußt",
                     "möchte", "mögen", "möglich", "mögt", "müssen", "müsst", "müßt", "n", "na", "nach", "nachdem",
                     "nahm",
                     "natürlich", "neben", "nein", "neue", "neuen", "neun", "neunte", "neunten", "neunter", "neuntes",
                     "nicht", "nichts", "nie", "niemand", "niemandem", "niemanden", "noch", "nun", "nur", "o", "ob",
                     "oben", "oder", "offen", "oft", "ohne", "ordnung", "p", "q", "r", "recht", "rechte", "rechten",
                     "rechter", "rechtes", "richtig", "rund", "s", "sa", "sache", "sagt", "sagte", "sah", "satt",
                     "schlecht", "schluss", "schon", "sechs", "sechste", "sechsten", "sechster", "sechstes", "sehr",
                     "sei",
                     "seid", "seien", "sein", "seine", "seinem", "seinen", "seiner", "seines", "seit", "seitdem",
                     "selbst",
                     "sich", "sie", "sieben", "siebente", "siebenten", "siebenter", "siebentes", "sind", "so", "solang",
                     "solche", "solchem", "solchen", "solcher", "solches", "soll", "sollen", "sollst", "sollt",
                     "sollte",
                     "sollten", "sondern", "sonst", "soweit", "sowie", "später", "startseite", "statt", "steht",
                     "suche",
                     "t", "tag", "tage", "tagen", "tat", "teil", "tel", "tritt", "trotzdem", "tun", "u", "uhr", "um",
                     "und", "uns", "unse", "unsem", "unsen", "unser", "unsere", "unserer", "unses", "unter", "v",
                     "vergangenen", "viel", "viele", "vielem", "vielen", "vielleicht", "vier", "vierte", "vierten",
                     "vierter", "viertes", "vom", "von", "vor", "w", "wahr", "wann", "war", "waren", "warst", "wart",
                     "warum", "was", "weg", "wegen", "weil", "weit", "weiter", "weitere", "weiteren", "weiteres",
                     "welche",
                     "welchem", "welchen", "welcher", "welches", "wem", "wen", "wenig", "wenige", "weniger", "weniges",
                     "wenigstens", "wenn", "wer", "werde", "werden", "werdet", "weshalb", "wessen", "wie", "wieder",
                     "wieso", "will", "willst", "wir", "wird", "wirklich", "wirst", "wissen", "wo", "woher", "wohin",
                     "wohl", "wollen", "wollt", "wollte", "wollten", "worden", "wurde", "wurden", "während",
                     "währenddem",
                     "währenddessen", "wäre", "würde", "würden", "x", "y", "z", "z.b", "zehn", "zehnte", "zehnten",
                     "zehnter", "zehntes", "zeit", "zu", "zuerst", "zugleich", "zum", "zunächst", "zur", "zurück",
                     "zusammen", "zwanzig", "zwar", "zwei", "zweite", "zweiten", "zweiter", "zweites", "zwischen",
                     "zwölf",
                     "über", "überhaupt", "übrigens"}

stop_word_list_fr = {"a", "abord", "absolument", "afin", "ah", "ai", "aie", "aient", "aies", "ailleurs", "ainsi", "ait",
                     "allaient", "allo", "allons", "allô", "alors", "anterieur", "anterieure", "anterieures", "apres",
                     "après", "as", "assez", "attendu", "au", "aucun", "aucune", "aucuns", "aujourd", "aujourd'hui",
                     "aupres", "auquel", "aura", "aurai", "auraient", "aurais", "aurait", "auras", "aurez", "auriez",
                     "aurions", "aurons", "auront", "aussi", "autant", "autre", "autrefois", "autrement", "autres",
                     "autrui", "aux", "auxquelles", "auxquels", "avaient", "avais", "avait", "avant", "avec", "avez",
                     "aviez", "avions", "avoir", "avons", "ayant", "ayez", "ayons", "b", "bah", "bas", "basee", "bat",
                     "beau", "beaucoup", "bien", "bigre", "bon", "boum", "bravo", "brrr", "c", "car", "ce", "ceci",
                     "cela", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là", "celui", "celui-ci",
                     "celui-là", "celà", "cent", "cependant", "certain", "certaine", "certaines", "certains", "certes",
                     "ces", "cet", "cette", "ceux", "ceux-ci", "ceux-là", "chacun", "chacune", "chaque", "cher",
                     "chers", "chez", "chiche", "chut", "chère", "chères", "ci", "cinq", "cinquantaine", "cinquante",
                     "cinquantième", "cinquième", "clac", "clic", "combien", "comme", "comment", "comparable",
                     "comparables", "compris", "concernant", "contre", "couic", "crac", "d", "da", "dans", "de",
                     "debout", "dedans", "dehors", "deja", "delà", "depuis", "dernier", "derniere", "derriere",
                     "derrière", "des", "desormais", "desquelles", "desquels", "dessous", "dessus", "deux", "deuxième",
                     "deuxièmement", "devant", "devers", "devra", "devrait", "different", "differentes", "differents",
                     "différent", "différente", "différentes", "différents", "dire", "directe", "directement", "dit",
                     "dite", "dits", "divers", "diverse", "diverses", "dix", "dix-huit", "dix-neuf", "dix-sept",
                     "dixième", "doit", "doivent", "donc", "dont", "dos", "douze", "douzième", "dring", "droite", "du",
                     "duquel", "durant", "dès", "début", "désormais", "e", "effet", "egale", "egalement", "egales",
                     "eh", "elle", "elle-même", "elles", "elles-mêmes", "en", "encore", "enfin", "entre", "envers",
                     "environ", "es", "essai", "est", "et", "etant", "etc", "etre", "eu", "eue", "eues", "euh",
                     "eurent", "eus", "eusse", "eussent", "eusses", "eussiez", "eussions", "eut", "eux", "eux-mêmes",
                     "exactement", "excepté", "extenso", "exterieur", "eûmes", "eût", "eûtes", "f", "fais", "faisaient",
                     "faisant", "fait", "faites", "façon", "feront", "fi", "flac", "floc", "fois", "font", "force",
                     "furent", "fus", "fusse", "fussent", "fusses", "fussiez", "fussions", "fut", "fûmes", "fût",
                     "fûtes", "g", "gens", "h", "ha", "haut", "hein", "hem", "hep", "hi", "ho", "holà", "hop", "hormis",
                     "hors", "hou", "houp", "hue", "hui", "huit", "huitième", "hum", "hurrah", "hé", "hélas", "i",
                     "ici", "il", "ils", "importe", "j", "je", "jusqu", "jusque", "juste", "k", "l", "la", "laisser",
                     "laquelle", "las", "le", "lequel", "les", "lesquelles", "lesquels", "leur", "leurs", "longtemps",
                     "lors", "lorsque", "lui", "lui-meme", "lui-même", "là", "lès", "m", "ma", "maint", "maintenant",
                     "mais", "malgre", "malgré", "maximale", "me", "meme", "memes", "merci", "mes", "mien", "mienne",
                     "miennes", "miens", "mille", "mince", "mine", "minimale", "moi", "moi-meme", "moi-même",
                     "moindres", "moins", "mon", "mot", "moyennant", "multiple", "multiples", "même", "mêmes", "n",
                     "na", "naturel", "naturelle", "naturelles", "ne", "neanmoins", "necessaire", "necessairement",
                     "neuf", "neuvième", "ni", "nombreuses", "nombreux", "nommés", "non", "nos", "notamment", "notre",
                     "nous", "nous-mêmes", "nouveau", "nouveaux", "nul", "néanmoins", "nôtre", "nôtres", "o", "oh",
                     "ohé", "ollé", "olé", "on", "ont", "onze", "onzième", "ore", "ou", "ouf", "ouias", "oust", "ouste",
                     "outre", "ouvert", "ouverte", "ouverts", "o|", "où", "p", "paf", "pan", "par", "parce", "parfois",
                     "parle", "parlent", "parler", "parmi", "parole", "parseme", "partant", "particulier",
                     "particulière", "particulièrement", "pas", "passé", "pendant", "pense", "permet", "personne",
                     "personnes", "peu", "peut", "peuvent", "peux", "pff", "pfft", "pfut", "pif", "pire", "pièce",
                     "plein", "plouf", "plupart", "plus", "plusieurs", "plutôt", "possessif", "possessifs", "possible",
                     "possibles", "pouah", "pour", "pourquoi", "pourrais", "pourrait", "pouvait", "prealable",
                     "precisement", "premier", "première", "premièrement", "pres", "probable", "probante", "procedant",
                     "proche", "près", "psitt", "pu", "puis", "puisque", "pur", "pure", "q", "qu", "quand", "quant",
                     "quant-à-soi", "quanta", "quarante", "quatorze", "quatre", "quatre-vingt", "quatrième",
                     "quatrièmement", "que", "quel", "quelconque", "quelle", "quelles", "quelqu'un", "quelque",
                     "quelques", "quels", "qui", "quiconque", "quinze", "quoi", "quoique", "r", "rare", "rarement",
                     "rares", "relative", "relativement", "remarquable", "rend", "rendre", "restant", "reste",
                     "restent", "restrictif", "retour", "revoici", "revoilà", "rien", "s", "sa", "sacrebleu", "sait",
                     "sans", "sapristi", "sauf", "se", "sein", "seize", "selon", "semblable", "semblaient", "semble",
                     "semblent", "sent", "sept", "septième", "sera", "serai", "seraient", "serais", "serait", "seras",
                     "serez", "seriez", "serions", "serons", "seront", "ses", "seul", "seule", "seulement", "si",
                     "sien", "sienne", "siennes", "siens", "sinon", "six", "sixième", "soi", "soi-même", "soient",
                     "sois", "soit", "soixante", "sommes", "son", "sont", "sous", "souvent", "soyez", "soyons",
                     "specifique", "specifiques", "speculatif", "stop", "strictement", "subtiles", "suffisant",
                     "suffisante", "suffit", "suis", "suit", "suivant", "suivante", "suivantes", "suivants", "suivre",
                     "sujet", "superpose", "sur", "surtout", "t", "ta", "tac", "tandis", "tant", "tardive", "te", "tel",
                     "telle", "tellement", "telles", "tels", "tenant", "tend", "tenir", "tente", "tes", "tic", "tien",
                     "tienne", "tiennes", "tiens", "toc", "toi", "toi-même", "ton", "touchant", "toujours", "tous",
                     "tout", "toute", "toutefois", "toutes", "treize", "trente", "tres", "trois", "troisième",
                     "troisièmement", "trop", "très", "tsoin", "tsouin", "tu", "té", "u", "un", "une", "unes",
                     "uniformement", "unique", "uniques", "uns", "v", "va", "vais", "valeur", "vas", "vers", "via",
                     "vif", "vifs", "vingt", "vivat", "vive", "vives", "vlan", "voici", "voie", "voient", "voilà",
                     "voire", "vont", "vos", "votre", "vous", "vous-mêmes", "vu", "vé", "vôtre", "vôtres", "w", "x",
                     "y", "z", "zut", "à", "â", "ça", "ès", "étaient", "étais", "était", "étant", "état", "étiez",
                     "étions", "été", "étée", "étées", "étés", "êtes", "être", "ô"}

stop_word_list_it = {"a", "abbastanza", "abbia", "abbiamo", "abbiano", "abbiate", "accidenti", "ad", "adesso",
                     "affinché", "agl", "agli", "ahime", "ahimè", "ai", "al", "alcuna", "alcuni", "alcuno", "all",
                     "alla", "alle", "allo", "allora", "altre", "altri", "altrimenti", "altro", "altrove", "altrui",
                     "anche", "ancora", "anni", "anno", "ansa", "anticipo", "assai", "attesa", "attraverso", "avanti",
                     "avemmo", "avendo", "avente", "aver", "avere", "averlo", "avesse", "avessero", "avessi",
                     "avessimo", "aveste", "avesti", "avete", "aveva", "avevamo", "avevano", "avevate", "avevi",
                     "avevo", "avrai", "avranno", "avrebbe", "avrebbero", "avrei", "avremmo", "avremo", "avreste",
                     "avresti", "avrete", "avrà", "avrò", "avuta", "avute", "avuti", "avuto", "basta", "ben", "bene",
                     "benissimo", "brava", "bravo", "buono", "c", "caso", "cento", "certa", "certe", "certi", "certo",
                     "che", "chi", "chicchessia", "chiunque", "ci", "ciascuna", "ciascuno", "cima", "cinque", "cio",
                     "cioe", "cioè", "circa", "citta", "città", "ciò", "co", "codesta", "codesti", "codesto", "cogli",
                     "coi", "col", "colei", "coll", "coloro", "colui", "come", "cominci", "comprare", "comunque", "con",
                     "concernente", "conclusione", "consecutivi", "consecutivo", "consiglio", "contro", "cortesia",
                     "cos", "cosa", "cosi", "così", "cui", "d", "da", "dagl", "dagli", "dai", "dal", "dall", "dalla",
                     "dalle", "dallo", "dappertutto", "davanti", "degl", "degli", "dei", "del", "dell", "della",
                     "delle", "dello", "dentro", "detto", "deve", "devo", "di", "dice", "dietro", "dire", "dirimpetto",
                     "diventa", "diventare", "diventato", "dopo", "doppio", "dov", "dove", "dovra", "dovrà", "dovunque",
                     "due", "dunque", "durante", "e", "ebbe", "ebbero", "ebbi", "ecc", "ecco", "ed", "effettivamente",
                     "egli", "ella", "entrambi", "eppure", "era", "erano", "eravamo", "eravate", "eri", "ero",
                     "esempio", "esse", "essendo", "esser", "essere", "essi", "ex", "fa", "faccia", "facciamo",
                     "facciano", "facciate", "faccio", "facemmo", "facendo", "facesse", "facessero", "facessi",
                     "facessimo", "faceste", "facesti", "faceva", "facevamo", "facevano", "facevate", "facevi",
                     "facevo", "fai", "fanno", "farai", "faranno", "fare", "farebbe", "farebbero", "farei", "faremmo",
                     "faremo", "fareste", "faresti", "farete", "farà", "farò", "fatto", "favore", "fece", "fecero",
                     "feci", "fin", "finalmente", "finche", "fine", "fino", "forse", "forza", "fosse", "fossero",
                     "fossi", "fossimo", "foste", "fosti", "fra", "frattempo", "fu", "fui", "fummo", "fuori", "furono",
                     "futuro", "generale", "gente", "gia", "giacche", "giorni", "giorno", "giu", "già", "gli", "gliela",
                     "gliele", "glieli", "glielo", "gliene", "grande", "grazie", "gruppo", "ha", "haha", "hai", "hanno",
                     "ho", "i", "ie", "ieri", "il", "improvviso", "in", "inc", "indietro", "infatti", "inoltre",
                     "insieme", "intanto", "intorno", "invece", "io", "l", "la", "lasciato", "lato", "le", "lei", "li",
                     "lo", "lontano", "loro", "lui", "lungo", "luogo", "là", "ma", "macche", "magari", "maggior", "mai",
                     "male", "malgrado", "malissimo", "me", "medesimo", "mediante", "meglio", "meno", "mentre", "mesi",
                     "mezzo", "mi", "mia", "mie", "miei", "mila", "miliardi", "milioni", "minimi", "mio", "modo",
                     "molta", "molti", "moltissimo", "molto", "momento", "mondo", "ne", "negl", "negli", "nei", "nel",
                     "nell", "nella", "nelle", "nello", "nemmeno", "neppure", "nessun", "nessuna", "nessuno", "niente",
                     "no", "noi", "nome", "non", "nondimeno", "nonostante", "nonsia", "nostra", "nostre", "nostri",
                     "nostro", "novanta", "nove", "nulla", "nuovi", "nuovo", "o", "od", "oggi", "ogni", "ognuna",
                     "ognuno", "oltre", "oppure", "ora", "ore", "osi", "ossia", "ottanta", "otto", "paese", "parecchi",
                     "parecchie", "parecchio", "parte", "partendo", "peccato", "peggio", "per", "perche", "perchè",
                     "perché", "percio", "perciò", "perfino", "pero", "persino", "persone", "però", "piedi", "pieno",
                     "piglia", "piu", "piuttosto", "più", "po", "pochissimo", "poco", "poi", "poiche", "possa",
                     "possedere", "posteriore", "posto", "potrebbe", "preferibilmente", "presa", "press", "prima",
                     "primo", "principalmente", "probabilmente", "promesso", "proprio", "puo", "pure", "purtroppo",
                     "può", "qua", "qualche", "qualcosa", "qualcuna", "qualcuno", "quale", "quali", "qualunque",
                     "quando", "quanta", "quante", "quanti", "quanto", "quantunque", "quarto", "quasi", "quattro",
                     "quel", "quella", "quelle", "quelli", "quello", "quest", "questa", "queste", "questi", "questo",
                     "qui", "quindi", "quinto", "realmente", "recente", "recentemente", "registrazione", "relativo",
                     "riecco", "rispetto", "salvo", "sara", "sarai", "saranno", "sarebbe", "sarebbero", "sarei",
                     "saremmo", "saremo", "sareste", "saresti", "sarete", "sarà", "sarò", "scola", "scopo", "scorso",
                     "se", "secondo", "seguente", "seguito", "sei", "sembra", "sembrare", "sembrato", "sembrava",
                     "sembri", "sempre", "senza", "sette", "si", "sia", "siamo", "siano", "siate", "siete", "sig",
                     "solito", "solo", "soltanto", "sono", "sopra", "soprattutto", "sotto", "spesso", "sta", "stai",
                     "stando", "stanno", "starai", "staranno", "starebbe", "starebbero", "starei", "staremmo",
                     "staremo", "stareste", "staresti", "starete", "starà", "starò", "stata", "state", "stati", "stato",
                     "stava", "stavamo", "stavano", "stavate", "stavi", "stavo", "stemmo", "stessa", "stesse",
                     "stessero", "stessi", "stessimo", "stesso", "steste", "stesti", "stette", "stettero", "stetti",
                     "stia", "stiamo", "stiano", "stiate", "sto", "su", "sua", "subito", "successivamente",
                     "successivo", "sue", "sugl", "sugli", "sui", "sul", "sull", "sulla", "sulle", "sullo", "suo",
                     "suoi", "tale", "tali", "talvolta", "tanto", "te", "tempo", "terzo", "th", "ti", "titolo", "tra",
                     "tranne", "tre", "trenta", "triplo", "troppo", "trovato", "tu", "tua", "tue", "tuo", "tuoi",
                     "tutta", "tuttavia", "tutte", "tutti", "tutto", "uguali", "ulteriore", "ultimo", "un", "una",
                     "uno", "uomo", "va", "vai", "vale", "vari", "varia", "varie", "vario", "verso", "vi", "vicino",
                     "visto", "vita", "voi", "volta", "volte", "vostra", "vostre", "vostri", "vostro", "è"}

stop_word_list_nl = {"aan", "aangaande", "aangezien", "achte", "achter", "achterna", "af", "afgelopen", "al", "aldaar",
                     "aldus", "alhoewel", "alias", "alle", "allebei", "alleen", "alles", "als", "alsnog", "altijd",
                     "altoos", "ander", "andere", "anders", "anderszins", "beetje", "behalve", "behoudens", "beide",
                     "beiden", "ben", "beneden", "bent", "bepaald", "betreffende", "bij", "bijna", "bijv", "binnen",
                     "binnenin", "blijkbaar", "blijken", "boven", "bovenal", "bovendien", "bovengenoemd", "bovenstaand",
                     "bovenvermeld", "buiten", "bv", "daar", "daardoor", "daarheen", "daarin", "daarna", "daarnet",
                     "daarom", "daarop", "daaruit", "daarvanlangs", "dan", "dat", "de", "deden", "deed", "der", "derde",
                     "derhalve", "dertig", "deze", "dhr", "die", "dikwijls", "dit", "doch", "doe", "doen", "doet",
                     "door", "doorgaand", "drie", "duizend", "dus", "echter", "een", "eens", "eer", "eerdat", "eerder",
                     "eerlang", "eerst", "eerste", "eigen", "eigenlijk", "elk", "elke", "en", "enig", "enige",
                     "enigszins", "enkel", "er", "erdoor", "erg", "ergens", "etc", "etcetera", "even", "eveneens",
                     "evenwel", "gauw", "ge", "gedurende", "geen", "gehad", "gekund", "geleden", "gelijk", "gemoeten",
                     "gemogen", "genoeg", "geweest", "gewoon", "gewoonweg", "haar", "haarzelf", "had", "hadden", "hare",
                     "heb", "hebben", "hebt", "hedden", "heeft", "heel", "hem", "hemzelf", "hen", "het", "hetzelfde",
                     "hier", "hierbeneden", "hierboven", "hierin", "hierna", "hierom", "hij", "hijzelf", "hoe",
                     "hoewel", "honderd", "hun", "hunne", "ieder", "iedere", "iedereen", "iemand", "iets", "ik",
                     "ikzelf", "in", "inderdaad", "inmiddels", "intussen", "inzake", "is", "ja", "je", "jezelf", "jij",
                     "jijzelf", "jou", "jouw", "jouwe", "juist", "jullie", "kan", "klaar", "kon", "konden", "krachtens",
                     "kun", "kunnen", "kunt", "laatst", "later", "liever", "lijken", "lijkt", "maak", "maakt", "maakte",
                     "maakten", "maar", "mag", "maken", "me", "meer", "meest", "meestal", "men", "met", "mevr",
                     "mezelf", "mij", "mijn", "mijnent", "mijner", "mijzelf", "minder", "miss", "misschien", "missen",
                     "mits", "mocht", "mochten", "moest", "moesten", "moet", "moeten", "mogen", "mr", "mrs", "mw", "na",
                     "naar", "nadat", "nam", "namelijk", "nee", "neem", "negen", "nemen", "nergens", "net", "niemand",
                     "niet", "niets", "niks", "noch", "nochtans", "nog", "nogal", "nooit", "nu", "nv", "of", "ofschoon",
                     "om", "omdat", "omhoog", "omlaag", "omstreeks", "omtrent", "omver", "ondanks", "onder",
                     "ondertussen", "ongeveer", "ons", "onszelf", "onze", "onzeker", "ooit", "ook", "op", "opnieuw",
                     "opzij", "over", "overal", "overeind", "overige", "overigens", "paar", "pas", "per", "precies",
                     "recent", "redelijk", "reeds", "rond", "rondom", "samen", "sedert", "sinds", "sindsdien",
                     "slechts", "sommige", "spoedig", "steeds", "tamelijk", "te", "tegen", "tegenover", "tenzij",
                     "terwijl", "thans", "tien", "tiende", "tijdens", "tja", "toch", "toe", "toen", "toenmaals",
                     "toenmalig", "tot", "totdat", "tussen", "twee", "tweede", "u", "uit", "uitgezonderd", "uw", "vaak",
                     "vaakwat", "van", "vanaf", "vandaan", "vanuit", "vanwege", "veel", "veeleer", "veertig", "verder",
                     "verscheidene", "verschillende", "vervolgens", "via", "vier", "vierde", "vijf", "vijfde",
                     "vijftig", "vol", "volgend", "volgens", "voor", "vooraf", "vooral", "vooralsnog", "voorbij",
                     "voordat", "voordezen", "voordien", "voorheen", "voorop", "voorts", "vooruit", "vrij", "vroeg",
                     "waar", "waarom", "waarschijnlijk", "wanneer", "want", "waren", "was", "wat", "we", "wederom",
                     "weer", "weg", "wegens", "weinig", "wel", "weldra", "welk", "welke", "werd", "werden", "werder",
                     "wezen", "whatever", "wie", "wiens", "wier", "wij", "wijzelf", "wil", "wilden", "willen", "word",
                     "worden", "wordt", "zal", "ze", "zei", "zeker", "zelf", "zelfde", "zelfs", "zes", "zeven", "zich",
                     "zichzelf", "zij", "zijn", "zijne", "zijzelf", "zo", "zoals", "zodat", "zodra", "zonder", "zou",
                     "zouden", "zowat", "zulk", "zulke", "zullen", "zult"}