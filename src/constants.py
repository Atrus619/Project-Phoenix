class Constants:
    # List of IP Vanish servers: https://account.ipvanish.com/index.php?t=Server%20List
    # TODO: Jen: Expand list later
    ipvanish_base_links = [
        ('iad-a01.ipvanish.com', 70),
        ('jnb-c01.ipvanish.com', 7)
    ]

    # List of user agents: https://developers.whatismybrowser.com/useragents/explore/
    # TODO: Jen: Expand list later
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36',
        'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322)',
        'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Windows NT 5.1; rv:7.0.1) Gecko/20100101 Firefox/7.0.1',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.1'
    ]

    base_request_headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://google.com'
    }

    page_break = '---------------------------------------------------------------------------------------'

    skills = {
        "Math": "Math",
        "Computer Science": "Computer Science",
        "Statistics": "Statistics",
        "Communication": "Communication",
        "Written": "Written Skills",
        "Verbal": "Verbal Skills",
        "natural language": "NLP",
        "NLP": "NLP",
        "Deep Learning": "Deep Learning",
        "Tensorflow": "Tensorflow",
        "Time Series": "Time Series",
        "Operations": "Operations",
        "presentation": "Presentation Skills",
        "analytics": "Analytics",
        "relationships": "Developing Relationships",
        "team": "Team Player",
        "SQL": "SQL",
        "VBA": "VBA",
        "Python": "Python",
        "Excel": "Excel",
        "Microsoft Office": "Microsoft Office"
    }