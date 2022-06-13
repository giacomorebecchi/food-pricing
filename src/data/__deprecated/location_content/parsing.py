from lxml import html


def parse_location_response(response_content: bytes, data_xpath: str) -> bytes:
    dom = html.fromstring(response_content)
    l = dom.xpath(data_xpath)
    if l:
        data = l[0].xpath("text()")
        if data:
            return data[0]
    return b""
