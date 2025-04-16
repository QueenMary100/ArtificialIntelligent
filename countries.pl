%Facts
capital(china, beijing).
capital(egypt, cairo).
capital(ethiopia, addis_ababa).
capital(france, paris).
capital(namibia, windhoek).
capital(germany, berlin).
capital(ghana, accra).
capital(senegal, dakar).

start :- write('What country do you want to know the capital?'),nl,
write('Type a country in lowercase followed by a period.'),nl,
read(A),
capital(A, B), write(B), write('   is the capital of   '), write(A),
write('.'), nl.
