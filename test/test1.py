import json
import requests
from bs4 import BeautifulSoup
import re
import validators
url = 'https://news.ycombinator.com/'
pattern = r'<a href="(.*?)"(?:\srel="nofollow noreferrer")?'
resp = requests.get(url)
soup = BeautifulSoup(resp.text, 'html.parser')
links = []
for _, item in enumerate(soup.find_all(class_='titleline')):
    link = re.findall(pattern, str(item))
    if len(link) > 0 and validators.url(link[0]):
        print(link[0])
        links.append(link[0])
print(links)
'''
https://ai.meta.com/blog/audiocraft-musicgen-audiogen-encodec-generative-ai-audio/
https://chargebackstop.com/blog/card-networks-exploitation/
https://praveshkoirala.com/2023/06/13/a-non-mathematical-introduction-to-kalman-filters-for-programmers/
https://aaronhertzmann.com/2020/04/19/lines-as-edges.html
https://catskull.net/html.html
https://www.decisionproblem.com/paperclips/index2.html
https://jarv.is/notes/cloudflare-dns-archive-is-blocked/
https://openjdk.org/projects/leyden/notes/03-toward-condensers
https://github.com/DTolm/VkFFT
https://github.com/reflex-dev/reflex
https://www.smithsonianmag.com/smart-news/when-beetle-gets-eaten-frog-it-forces-its-way-out-back-door-180975484/
https://austinhenley.com/blog/90percent.html
https://eugeneyan.com/writing/llm-patterns/
https://bigthink.com/the-learning-curve/the-onion-founder-strategies-sparking-creativity/
https://oxocard.ch/en/
https://v5.chriskrycho.com/journal/unsafe/
https://max.levch.in/post/724289457144070144/shamir-secret-sharing
https://eol.org/
https://longform.asmartbear.com/problem/
https://devblogs.microsoft.com/oldnewthing/20230802-00/?p=108524
https://bitmovin.com/careers/
https://www.smithsonianmag.com/history/how-the-kentucky-cave-wars-reshaped-the-states-tourism-industry-180982585/
https://hackaday.com/2023/07/30/an-open-source-free-circuit-simulator/
https://arxiv.org/abs/2308.00676
https://www.overshootday.org/about/
https://vienna.earth/plate/russell/kafka-insurance-career
https://www.apple.com/newsroom/2023/08/pixar-adobe-apple-autodesk-and-nvidia-form-alliance-for-openusd/
https://www.businessinsider.com/harvard-francesca-gino-fake-data-fraud-allegations-scholar-2023-7
'''