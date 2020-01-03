"""

https://nitratine.net/blog/post/encryption-and-decryption-in-python/



http://www.blog.pythonlibrary.org/2016/05/18/python-3-an-intro-to-encryption/

The cryptography package aims to be “cryptography for humans” much like the requests library is “HTTP for Humans”. The idea is that you will be able to create simple cryptographic recipes that are safe and easy-to-use. If you need to, you can drop down to low=level cryptographic primitives, which require you to know what you’re doing or you might end up creating something that’s not very secure.

If you are using Python 3.5, you can install it with pip, like so:


pip install cryptography

You will see that cryptography installs a few dependencies along with itself. Assuming that they all completed successfully, we can try encrypting some text. Let’s give the Fernet module a try. The Fernet module implements an easy-to-use authentication scheme that uses a symmetric encryption algorithm which guarantees that any message you encrypt with it cannot be manipulated or read without the key you define. The Fernet module also supports key rotation via MultiFernet. Let’s take a look at a simple example:

        >>> from cryptography.fernet import Fernet
        >>> cipher_key = Fernet.generate_key()
        >>> cipher_key
        b'APM1JDVgT8WDGOWBgQv6EIhvxl4vDYvUnVdg-Vjdt0o='
        >>> cipher = Fernet(cipher_key)
        >>> text = b'My super secret message'
        >>> encrypted_text = cipher.encrypt(text)
        >>> encrypted_text
        (b'gAAAAABXOnV86aeUGADA6mTe9xEL92y_m0_TlC9vcqaF6NzHqRKkjEqh4d21PInEP3C9HuiUkS9f'
         b'6bdHsSlRiCNWbSkPuRd_62zfEv3eaZjJvLAm3omnya8=')
        >>> decrypted_text = cipher.decrypt(encrypted_text)
        >>> decrypted_text
        b'My super secret message'
        First off we need to import Fernet. Next we generate a key. We print out the key to see what it looks like. As you can see, it’s a random byte string. If you want, you can try running the generate_key method a few times. The result will always be different. Next we create our Fernet cipher instance using our key.

Now we have a cipher we can use to encrypt and decrypt our message. The next step is to create a message worth encrypting and then encrypt it using the encrypt method. I went ahead and printed our the encrypted text so you can see that you can no longer read the text. To decrypt our super secret message, we just call decrypt on our cipher and pass it the encrypted text. The result is we get a plain text byte string of our message.

"""

import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pathlib import Path
import re

DIVIDER = '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'.encode('utf-8')

class SFile():
    def __init__(self, file_name, salt_file='prod_key'):
        """
        Create file buffer and read/write encrypted lines to and from it. E.g.

        # abc123
        sf = SFile('/somewhere/somefile.bin')
        for i in range(10):
            sf.append(f'More stuf fShort message number {i} of 10; '*i)
        print(sf.read())


        :param file_name: fully qualified filename
        :param password:
        :param salt_file:
        :return:
        """
        password = input('Password: ')
        self.file_name = file_name
        self.file = Path(file_name)
        self.file.parent.mkdir(parents=True, exist_ok=True)
        if not self.file.exists():
            self.file.touch()
        self.salt = self.read_salt(salt_file)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
            backend=default_backend()
        )  # can only use once
        self.key = base64.urlsafe_b64encode(kdf.derive(password.encode('utf-8')))
        # OK, ready for action

    def append(self, txt):
        """
        Append txt to file
        :param txt:
        :return:
        """
        f = Fernet(self.key)
        btxt = txt.encode('utf-8')
        token = f.encrypt(btxt)
        with self.file.open('ab') as f:
            f.write(DIVIDER)
            f.write(token)

        # f.decrypt( f.encrypt('messasdfasdfage'.encode('utf-8')) ).decode('utf-8')

    def read(self):
        """
        read and decrypt file

        :return:
        """


        with self.file.open('rb') as ff:
            b = ff.read()

        f = Fernet(self.key)
        out = []
        for token in re.split(DIVIDER, b)[1:]:
            out.append(f.decrypt(token).decode('utf-8'))

        return '\n'.join(out)

    @staticmethod
    def new_salt(fn, l):
        p = Path.home() / f'Documents/.salt/{fn}.skey'
        p.parent.mkdir(parents=True, exist_ok=True)
        salt = os.urandom(l)
        with p.open('wb') as f:
            f.write(salt)
        return salt

    @staticmethod
    def read_salt(fn):
        p = Path.home() / f'Documents/.salt/{fn}.skey'
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            return SFile.new_salt(fn, 128)
        with p.open('rb') as f:
            salt = f.read()
        return salt



