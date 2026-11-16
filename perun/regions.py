NORTH_AMERICA = {"US", "CA", "PR", "BM"}
LATAM = {
    "MX", "BR", "AR", "CL", "CO", "PE", "EC", "VE", "UY", "PY", "BO", "CR", "PA", "GT",
    "HN", "SV", "NI", "DO", "CU", "JM", "TT", "BS", "BB", "BZ", "GY", "HT", "AG", "AW",
    "CW", "KY", "DM", "KN", "LC", "VC", "VG", "TC", "AI", "MS", "MQ",
}
EUROPE = {
    "GB", "DE", "FR", "IT", "ES", "PT", "NL", "BE", "LU", "CH", "AT", "IE", "DK", "SE",
    "NO", "FI", "IS", "PL", "CZ", "SK", "HU", "RO", "BG", "GR", "HR", "SI", "RS", "BA",
    "ME", "MK", "AL", "EE", "LV", "LT", "MT", "CY", "MC", "LI", "AD", "VA", "JE", "FO",
    "GL", "AX", "SJ", "UA", "MD", "TR",
}
CIS = {"RU", "BY", "KZ", "UZ", "KG", "TJ", "TM", "AM", "AZ", "GE"}
MEA = {
    "AE", "SA", "IL", "QA", "KW", "BH", "OM", "JO", "LB", "IQ", "IR", "SY", "YE", "PS",
    "EG", "MA", "DZ", "TN", "LY", "ZA", "NG", "KE", "GH", "ET", "TZ", "UG", "RW", "SN",
    "CI", "CM", "AO", "MZ", "ZM", "ZW", "BW", "NA", "MU", "SC", "MG", "CD", "CG", "GA",
    "BF", "BJ", "ML", "NE", "TD", "SD", "SO", "DJ", "GM", "GN", "SL", "LS", "MR", "RE", "YT",
}

REGIONS = {
    "Северная Америка": NORTH_AMERICA,
    "Латинская Америка": LATAM,
    "Европа": EUROPE,
    "СНГ": CIS,
    "Ближний Восток и Африка": MEA,
}


def region_of(iso2: str) -> str:
    if not iso2:
        return "Не указан"
    for name, codes in REGIONS.items():
        if iso2 in codes:
            return name
    return "Азия и Океания"
