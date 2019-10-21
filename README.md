# Mal2d: 2d based Deep Learning Model for Malware Detection using Black and White Binary Image

* Updated: We gather the information of KISA's malware files, our training dataset, through virusTotal.com

* Updated: We give additional experimental logs. (See additional_log/ )

* Updated: The log files have been added.

* Updated: The source files have been added.

* Updated: The data files have been uploaded. 

    * Data Files link : https://drive.google.com/open?id=1Jyi_5OAXt-FlBR9LtKKv1IzZmZYSULEW (It's about ~13GB) 

# Training Data # 

- **vir_sample_65536_kisa_2a.npz**   # black and white representation : 256 x (256 * 8) dimension

- **vir_sample_65536_kisa_2c.npz**    # gray scale representation : 256 x 256 dimension

# Testing Data #

## [malware only] ##

- **train_65536_ms_2a.npz**    # black and white representation : 256 x (256 * 8) dimension

- **train_65536_ms_2c.npz**    # gray scale representation : 256 x 256 dimension

## [benign only] ##

- **renameAll_65536_msorg_2a.npz**    # black and white representation : 256 x (256 * 8) dimension

- **renameAll_65536_msorg_2c.npz**    # gray scale representation : 256 x 256 dimension

Note : Due to the huge model size of VGG and ResNet, you may need to reduce their batch_size on your machine. (Recommended : ~24GB GPU memory)

If you have any questions, please email me ( minkcho@gmail.com ).
