; RECORDATORIOS
; inputs: rdi, rsi, rdx, rcx, r8, r9
; preservar: r12, r13, r14, r15, rbx, 
; la pila: rbp, rsp
; devolver cosas por rax o xmmo 
; inputs floats: xmm0, xmm1, ..., xmm7

	global cost_derivative

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
	push r12
	push r13
	push r14

	mov r12, rdi			;r12 = matrix
	mov r13, rsi			;r13 = matrix2
	mov r14, r8 			;r14 = output

	;Calculo la cantidad de pixeles total
	xor rax, rax
	mov eax, edx
	mul ecx					;eax = low(n*m) ;edx = high(n*m)
	shl rdx, 32
	add rax, rdx			;rax = #pixeles

	;Inicializo el contador
	mov rcx, rax
	shr rcx, 1				;Proceso de a 2 pixeles


	;Itero sobre todos los pixeles y realizo la operación de SUBPD
	.ciclo:
		movdqu xmm1, [r13]	;xmm1 = | px0 | px1 |
		movdqu xmm2, [r12]	;xmm2 = | px0'| px1'|

		SUBPD xmm1, xmm2

		movdqu [r14], xmm1

		;Avanzo los punteros
		add r12, 16
		add r13, 16
		add r14, 16
		loop .ciclo
	
	pop r14	
	pop r13
	pop r12	
	pop rbp
   	ret