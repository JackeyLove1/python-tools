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

### 