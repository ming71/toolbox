# -*- coding: utf-8 -*-
import scrapy


#class BooksSpider(scrapy.Spider):
#    name = "books"  # 每一个爬虫的唯一标识
#    start_urls = ['http://books.toscrape.com/'] # 定义爬虫爬取的起始点,起始点可以是多个,这里只有一个
#    
#    def parse(self, response):
#        # 提取数据
#        # 每一本书的信息在<article class="product_pod">中,使用css()方法找到所有这样的article元素,并依次迭代
#        # 语法也可选择xpath
#        for book in response.css('article.product_pod'):
#        # 书名信息在article > h3 > a 元素的title属性里
#        # 例如: <a title="A Light in the Attic">A Light in the ..
#            name = book.xpath('./h3/a/@title').extract_first()
#            # 书价信息在 <p class="price_color">的TEXT中。
#            # 例如: <p class="price_color">£51.77</p>
#            price = book.css('p.price_color::text').extract_first()
#            yield {
#            'name': name,
#            'price': price,
#            }
#        # 提取翻页链接
#        # 下一页的url 在ul.pager > li.next > a 里面
#        # 例如: <li class="next"><a href="catalogue/page-2.html">next
#        next_url = response.css('ul.pager li.next a::attr(href)').extract_first()
#        if next_url:
#        # 如果找到下一页的URL,得到绝对路径,构造新的Request 对象
#            next_url = response.urljoin(next_url)
#            yield scrapy.Request(next_url, callback=self.parse)            




class BooksSpider(scrapy.Spider):
    name = "books"  
    start_urls = ['https://newhouse.fang.com/house/s/'] 
    
    def parse(self, response):
        
        
        for i in range (1,22):
            next_url = 'https://newhouse.fang.com/house/s/b9'+str(i)+'/'
#            print(next_url)
#            a=input()
            for book in response.css('div.nlc_details'):
                name = book.xpath('.//div[@class="address"]/a/@title').extract_first()
                price = book.xpath('.//div[@class="nhouse_price"]/span/text()').extract_first()
                unit = book.xpath('.//div[@class="nhouse_price"]/em/text()').extract_first()
                yield {
                'name': name,
                'price': price,
                '单位': unit
                }
                
            next_url = response.urljoin(next_url)
            yield scrapy.Request(next_url, callback=self.parse) 
            
            
            
            
            
            
            
            
