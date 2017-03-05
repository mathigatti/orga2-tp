; RECORDATORIOS
; inputs: rdi, rsi, rdx, rcx, r8, r9
; preservar: r12, r13, r14, r15, rbx, 
; la pila: rbp, rsp
; devolver cosas por rax o xmmo 
; inputs floats: xmm0, xmm1, ..., xmm7

	global cost_derivative
	global mat_plus_vec

; YA IMPLEMENTADAS EN C
	extern fprintf
	extern malloc
	extern free
	extern fclose
	extern fopen

; /** DEFINES **/ 
	%define NULL 		0
	%define TRUE 		1
	%define FALSE 		0
	
	%define LF					10


section .rodata


section .data
	msg1: DB '%s', LF, 0	;imprimo string con salto de linea y fin de linea
	msg2: DB 'a',0			;modo append para fopen
	msg3: DB '<oracionVacia>',0
	

section .text


;/** FUNCIONES MATRICIALES **/
;-----------------------------------------------------------


; void cost_derivative(
;	double* matrix  (rdi)
;	double* matrix2 (rsi)
;	uint n 			(rdx)
;	uint m 			(rcx)
;	double* output	(r8)
; )

	cost_derivative:
	push rbp
	mov rbp, rsp

	;Calculo la cantidad de elementos total
	xor rax, rax
	mov eax, edx
	mul ecx					;eax = low(n*m) ;edx = high(n*m)
	shl rdx, 32
	add rax, rdx			;rax = #pixeles

	;Inicializo el contador
	mov rcx, rax
	shr rcx, 1				;Proceso de a 2 pixeles

	;Itero sobre todos los elementos y realizo la operación de SUBPD
	.ciclo:
		movupd xmm1, [rdi]	;xmm1 = | px0 | px1 |
		movupd xmm2, [rsi]	;xmm2 = | px0'| px1'|

		subpd xmm1, xmm2

		movupd [r8], xmm1

		;Avanzo los punteros
		add rdi, 16
		add rsi, 16
		add r8, 16
		loop .ciclo
	
	pop rbp
   	ret

	;void mat_plus_vec(
	;	double* matrix, (rdi) 
	; double* vector, (rsi)
	; uint n, (rdx)
	; uint m, (rcx)
	; double* output (r8)
	; )
	mat_plus_vec:
	push rbp
	mov rbp, rsp

	;Calculo la cantidad de pixeles total
	xor rax, rax
	mov eax, edx
	mul ecx					;eax = low(n*m) ;edx = high(n*m)
	shl rdx, 32
	add rax, rdx			;rax = #elementos

	;Chequeo si la cantidad de elementos es par
	xor rdx, rdx
	inc rdx
	and dl, al
	cmp dl, 0
	jz .even_case

	.odd_case:
	;Caso impar: opero sobre el primer elemento por separado
	movd xmm1, [rdi]
	movd xmm2, [rsi]
	addsd xmm1, xmm2
	movd [r8], xmm1
	add rdi, 8
	add rsi, 8
	add r8, 8	
	dec rax

	;Inicializo el contador
	.even_case:
	mov rcx, rax
	shr rcx, 1				;Proceso de a 2 pixeles, 

	;Itero sobre todos los pixeles y realizo la operación de SUBPD
	.ciclo:
		movupd xmm1, [rdi]	;xmm1 = | px0 | px1 |
		movupd xmm2, [rsi]	;xmm2 = | px0'| px1'|

		addpd xmm1, xmm2

		movupd [r8], xmm1

		;Avanzo los punteros
		add rdi, 16
		add rsi, 16
		add r8, 16
		loop .ciclo
	
	pop rbp
   	ret