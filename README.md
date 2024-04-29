# NanoGPT Fork 

Fork of [Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT)

### Differences:
- Combined the 'Head' and 'MultiHeadAttention' into one class to process all the heads in parallel. 
- Trained on a custom dataset. It is an old Filipino Poem named 'Ibong Adarna'.
- Used byte-pair encoding rather than the character-level encoding. 

### Sample Generation

```
  Nang sa princesang maquita
si don Juan niyang sintá,
himbing na ualang capara
ay guinising nang búhay,
baca ang quinuha
nang laroón na misariness.

  Cun ang uica nang haring amá
ang amyo poon cong amá.

  Saan patutungo naman
ang sa haring carunungan,
na májica negra lamang
ang caniyang tinatangnan.

  Ang sa cahoy sa caparangan
sampóng cantáng maiinam,
aco'i, huag malubay ca
sa iyo nama'i, panganay.

  Ay ano'i, nang maguising na
itong daquilang monarca,
ay nagtuloy sa lamesa
ang tinapay ay naquita.

  Nang damputin ang tinapay
ay napamatáy,

  Anó'i, nang maguiguing mayroó
ang caniyang haring amá.

```