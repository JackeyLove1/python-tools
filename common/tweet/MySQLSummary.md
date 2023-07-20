# MySQL Notes

## Basic

### DISTINCT

select DISTINCT vend_id from products;

### limit

only select 5 lines
select * from products limit 5;
select [5, 10) lines
select * from products limit 5, 5;

### order by

select prod_name from products order by prod_name;
descend order
select prod_name from products order by prod_name desc, prod_price;

### where

select prod_name, prod_price from products where prod_price=2.50;
not equal
select vend_id, prod_name from products where vend_id!=1003;
find data in [a,b]
select prod_name, prod_price from products where prod_price between 5 and 10;
null or not null
select prod_name, prod_price from products where prod_price is null;

### AND / OR

select prod_id, prod_price, prod_price from products where vend_id = 1003 and prod_price <= 10;
priority
select prod_id, prod_price, prod_price from products where (vend_id = 1002 or vend_id = 1003) and prod_price > 10;

### IN

select prod_id, prod_price, prod_price from products where (vend_id in (1002 , 1003)) and prod_price > 10;
IN (select ... )
IN WHERE

### NOT

select prod_id, prod_price, prod_price from products where (vend_id not in (1002 , 1003)) and prod_price > 10;
NOT WHERE

### LIKE

select prod_id, prod_name from products where prod_name like 'jet%';

## function

### AVG

select AVG(prod_price) AS avg_price from products where vend_id=1003;

### COUNT

select COUNT(*) AS num_customers from customers;

### MAX / MIN

### WHERE filter row and having filter group

select vend_id, COUNT(*) AS num_prods FROM products where prod_price>=10 group by vend_id having count(*) >= 2;

### 检索总计订单价格大于等于50的订单的订单号和总计订单价格

SELECT order_num, SUM(item_price * quantity) AS total_price
FROM orderitems
GROUP BY order_num
HAVING total_price >= 50;

### 联合查询：列出订购物品TNT2的所有客户

SELECT o.cust_id
FROM orders o
JOIN orderitems i ON o.order_num = i.order_num
WHERE i.prod_id = 'TNT2';
select cust_id from orders where order_num in (select order_num from orderitems where prod_id = 'TNT2');

### inner join 
查找二者都有的行
### left/right outer join
更具一边的行查找另一边与之匹配的行

## Operation
### insert 
insert into customers(...) values( ... );

### update
update customers set cust_email="example@123.com" where cust_id=15;

### create
CREATE TABLE orders
(
order_num int NOT NULL AUTO_INCREMENT,
order_date datetime NOT NULL,
cust_id int NOT NULL,
prod_id    char(10)      NOT NULL,
quantity   int           NOT NULL,
item_price decimal(8, 2) NOT NULL,
PRIMARY KEY(order_num)
)ENGINE=InnoDB;

### ALTER TABLE
ALTER TABLE vendors ADD vend_phone CHAR(20);
ALTER TABLE vendors DROP COLUMN vend_phone;

### delete table
DROP TABLE customer2;

### 存储过程Stored Procedure
CREATE PROCEDURE update_customer_balance (IN cust_id INT, IN amount DECIMAL(10,2))
BEGIN
    DECLARE current_balance DECIMAL(10,2);
    SELECT balance INTO current_balance
    FROM customers
    WHERE cust_id = cust_id;
    IF current_balance >= amount THEN
        UPDATE customers
        SET balance = balance - amount
        WHERE cust_id = cust_id;
    END IF;
END;

### transaction processing
