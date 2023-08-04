XPath（XML Path Language）是一种用于在XML文档中定位元素的语言。它可以通过路径表达式来选择XML文档中的节点。

XPath语法基于节点关系和节点属性，下面是一些XPath语法的基本要点：

1. 元素选择：
   - `element`：选择所有名称为"element"的元素节点。
   - `*`：选择所有元素节点。
   - `element1/element2`：选择所有属于"element1"的子元素"element2"的元素节点。

2. 路径表达式：
   - `//element`：选择文档中的所有"element"元素节点，不考虑它们的位置。
   - `/element1/element2`：选择顶层元素"element1"的子元素"element2"的元素节点。

3. 属性选择：
   - `element[@attribute='value']`：选择具有指定属性和值的元素节点。
   - `element[@attribute]`：选择具有指定属性的元素节点，无论其值如何。

4. 谓语（Predicates）：
   - `element[position()]`：选择具有指定位置的元素节点。
   - `element[@attribute='value'][position()]`：选择具有指定属性和值，并且在指定位置的元素节点。

5. 通配符：
   - `*`：匹配任意元素节点。
   - `@*`：匹配任意属性节点。

下面是一些XPath语法的例子：

1. 选择所有`div`元素：
   - XPath表达式：`//div`

2. 选择具有`class`属性为"example"的`div`元素：
   - XPath表达式：`//div[@class='example']`

3. 选择第一个`input`元素：
   - XPath表达式：`(//input)[1]`

4. 选择具有`name`属性的所有元素：
   - XPath表达式：`//*[@name]`

这些例子只是XPath语法的基本示例，XPath还有更多的功能和用法，可以用于更复杂的节点选择和过滤。

选取节点
XPath 使用路径表达式在 XML 文档中选取节点。节点是通过沿着路径或者 step 来选取的。 下面列出了最有用的路径表达式：

表达式	描述
nodename	选取此节点的所有子节点。
/	从根节点选取（取子节点）。
//	从匹配选择的当前节点选择文档中的节点，而不考虑它们的位置（取子孙节点）。
.	选取当前节点。
..	选取当前节点的父节点。
@	选取属性。
在下面的表格中，我们已列出了一些路径表达式以及表达式的结果：

路径表达式	结果
bookstore	选取 bookstore 元素的所有子节点。
/bookstore	
选取根元素 bookstore。

注释：假如路径起始于正斜杠( / )，则此路径始终代表到某元素的绝对路径！

bookstore/book	选取属于 bookstore 的子元素的所有 book 元素。
//book	选取所有 book 子元素，而不管它们在文档中的位置。
bookstore//book	选择属于 bookstore 元素的后代的所有 book 元素，而不管它们位于 bookstore 之下的什么位置。
//@lang	选取名为 lang 的所有属性。

谓语（Predicates）
谓语用来查找某个特定的节点或者包含某个指定的值的节点。

谓语被嵌在方括号中。

在下面的表格中，我们列出了带有谓语的一些路径表达式，以及表达式的结果：

路径表达式	结果
/bookstore/book[1]	选取属于 bookstore 子元素的第一个 book 元素。
/bookstore/book[last()]	选取属于 bookstore 子元素的最后一个 book 元素。
/bookstore/book[last()-1]	选取属于 bookstore 子元素的倒数第二个 book 元素。
/bookstore/book[position()<3]	选取最前面的两个属于 bookstore 元素的子元素的 book 元素。
//title[@lang]	选取所有拥有名为 lang 的属性的 title 元素。
//title[@lang='eng']	选取所有 title 元素，且这些元素拥有值为 eng 的 lang 属性。
/bookstore/book[price>35.00]	选取 bookstore 元素的所有 book 元素，且其中的 price 元素的值须大于 35.00。
/bookstore/book[price>35.00]//title	选取 bookstore 元素中的 book 元素的所有 title 元素，且其中的 price 元素的值须大于 35.00。

选取未知节点
XPath 通配符可用来选取未知的 XML 元素。

通配符	描述
*	匹配任何元素节点。
@*	匹配任何属性节点。
node()	匹配任何类型的节点。
在下面的表格中，我们列出了一些路径表达式，以及这些表达式的结果：

路径表达式	结果
/bookstore/*	选取 bookstore 元素的所有子元素。
//*	选取文档中的所有元素。
//title[@*]	选取所有带有属性的 title 元素。