exports.handler = async (event, context) => {
    return handleRequest(event.request)
}

async function handleRequest(request) {
    const PRESHARED_AUTH_HEADER_KEY = "X-Custom-PSK";
    const PRESHARED_AUTH_HEADER_VALUE = "your local key";
    const psk = request.headers.get(PRESHARED_AUTH_HEADER_KEY);

    if (psk !== PRESHARED_AUTH_HEADER_VALUE) {
        return new Response("Sorry, you have supplied an invalid key.", {status: 403,});
    }
    const url = new URL(request.url);
    const headers_Origin = request.headers.get("Access-Control-Allow-Origin") || "*"
    url.host = TELEGRAPH_URL.replace(/^https?:\/\//, '');

    // Copy headers except the pre-shared key
    let forwardHeaders = new Headers();
    for (let pair of request.headers.entries()) {
        if (pair[0].toLowerCase() !== PRESHARED_AUTH_HEADER_KEY.toLowerCase()) {
            forwardHeaders.append(pair[0], pair[1]);
        }
    }
    // proxy
    const modifiedRequest = new Request(url.toString(), {
        headers: forwardHeaders,
        method: request.method,
        body: request.body,
        redirect: 'follow'
    });
    const response = await fetch(modifiedRequest);
    const modifiedResponse = new Response(response.body, response);
    // 添加允许跨域访问的响应头
    modifiedResponse.headers.set('Access-Control-Allow-Origin', headers_Origin);
    return modifiedResponse;
}

