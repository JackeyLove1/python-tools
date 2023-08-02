from bs4 import BeautifulSoup
import re

html = """
<code class="hljs language-shell code-block-extension-codeShowNum" lang="shell"><span class="code-block-extension-codeLine" data-line-num="1"><span class="hljs-meta prompt_">&gt; </span><span class="bash">docker run busybox <span class="hljs-built_in">ls</span> -lh <span class="hljs-comment"># 运行标准的 unix 命令</span></span></span>
<span class="code-block-extension-codeLine" data-line-num="2"><span class="hljs-meta prompt_">&gt; </span><span class="bash">docker run &lt;image&gt;:&lt;tag&gt;  <span class="hljs-comment"># 运行指定版本的 image，tag 默认 latest</span></span></span>
<span class="code-block-extension-codeLine" data-line-num="3"><span class="hljs-meta prompt_"></span></span>
<span class="code-block-extension-codeLine" data-line-num="4"># <span class="bash">Dockerfile 包含构建 docker 镜像的命令</span></span>
<span class="code-block-extension-codeLine" data-line-num="5">FROM node # 基础镜像</span>
<span class="code-block-extension-codeLine" data-line-num="6">ADD app.js /app.js # 将本地文件添加到镜像的根目录</span>
<span class="code-block-extension-codeLine" data-line-num="7">ENTRYPOINT ["node", "app.js"] # 镜像被执行时需被执行的命令</span>
<span class="code-block-extension-codeLine" data-line-num="8"><span class="hljs-meta prompt_"></span></span>
<span class="code-block-extension-codeLine" data-line-num="9">&gt; <span class="bash">docker build -t kubia . <span class="hljs-comment"># 在当前目录根据 Dockerfile 构建指定 tag 的镜像</span></span></span>
<span class="code-block-extension-codeLine" data-line-num="10"><span class="hljs-meta prompt_">&gt; </span><span class="bash">docker images <span class="hljs-comment"># 列出本地所有镜像</span></span></span>
<span class="code-block-extension-codeLine" data-line-num="11"><span class="hljs-meta prompt_"></span></span>
<span class="code-block-extension-codeLine" data-line-num="12"># <span class="bash">执行基于 kubia 镜像，映射主机 8081 到容器内 8080 端口，并在后台运行的容器</span></span>
<span class="code-block-extension-codeLine" data-line-num="13"><span class="hljs-meta prompt_">&gt; </span><span class="bash">docker run --name kubia-container -p 8081:8080 -d kubia</span></span>
<span class="code-block-extension-codeLine" data-line-num="14"><span class="hljs-meta prompt_">&gt; </span><span class="bash">docker ps <span class="hljs-comment"># 列出 running 容器</span></span></span>
<span class="code-block-extension-codeLine" data-line-num="15"><span class="hljs-meta prompt_">&gt; </span><span class="bash">docker ps -a <span class="hljs-comment"># 列出 running, exited 容器</span></span></span>
<span class="code-block-extension-codeLine" data-line-num="16"><span class="hljs-meta prompt_"></span></span>
<span class="code-block-extension-codeLine" data-line-num="17">&gt; <span class="bash">docker <span class="hljs-built_in">exec</span> -it kubia-container bash <span class="hljs-comment"># 在容器内执行 shell 命令，如 ls/sh</span></span></span>
<span class="code-block-extension-codeLine" data-line-num="18"><span class="hljs-meta prompt_">&gt; </span><span class="bash">docker stop kubia-container <span class="hljs-comment"># 停止容器</span></span></span>
<span class="code-block-extension-codeLine" data-line-num="19"><span class="hljs-meta prompt_">&gt; </span><span class="bash">docker <span class="hljs-built_in">rm</span> kubia-container <span class="hljs-comment"># 删除容器</span></span></span>
<span class="code-block-extension-codeLine" data-line-num="20"><span class="hljs-meta prompt_">&gt; </span><span class="bash">docker tag kubia wuyinio/kubia <span class="hljs-comment"># 给本地镜像打标签</span></span></span>
<span class="code-block-extension-codeLine" data-line-num="21"><span class="hljs-meta prompt_"></span></span>
<span class="code-block-extension-codeLine" data-line-num="22">&gt; <span class="bash">docker login</span></span>
<span class="code-block-extension-codeLine" data-line-num="23"><span class="hljs-meta prompt_">&gt; </span><span class="bash">docker push wuyinio/kubia <span class="hljs-comment"># push 到 DockerHub</span></span></span>
</code>
"""

soup = BeautifulSoup(html, "html.parser")

# Find all span tags
span_tags = soup.find_all('span')
print(span_tags)

# List to hold span text
span_text_list = []

for tag in span_tags:
    tag_text = tag.text
    if re.search(r'[\u4e00-\u9fff]+', tag_text):
        # If Chinese characters are present, add to the list
        span_text_list.append(tag_text)

print(span_text_list)

# Write to text file
with open("output.txt", "w", encoding='utf-8') as file:
    for line in span_text_list:
        file.write(line + "\n")