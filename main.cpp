#include "GMM.h"
#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include "gdal_priv.h"  
#include "cpl_conv.h"

struct imageData;
class classify;
struct checkData;
classify* loadFile(std::string path, int &n);
void readData(classify* classifies, imageData* pData, Eigen::MatrixXd** trainMatrix,
	Eigen::MatrixXd** allMatrix);
Eigen::MatrixXi* classifyData(GMM** models, double **image, int nData, int nBand, int nClassify,
	int xSize, int ySize);
void outputImage(std::string fileName, Eigen::MatrixXi* resultMatrix);
void checkDataF(GMM** models, int nClassify, int nBand, checkData* pCheckData);
void outputCheck(checkData* pCheckData, std::string pathFile);
void outputImage2(std::string fileName, Eigen::MatrixXi* resultMatrix, int nClassify);

class classify
{
public:
	int m_n;//µ±ÖÐÓÐ¼¸¸öÇøÓò;
	int m_component;//¼ÇÂ¼Ã¿Ò»¸ö¸ßË¹»ìºÏÄ£ÐÍ·Ö¼¸¸öcomponent;
	int *m_up;
	int *m_down;
	int *m_left;
	int *m_right;//m_n¸öÊý£¬´ú±íÃ¿¸öÑµÁ·ÇøÓòµÄÎ»ÖÃ;
	void deleteData()
	{
		delete[]m_up;
		m_up = NULL;
		delete[]m_down;
		m_down = NULL;
		delete[]m_left;
		delete[]m_right;
		m_left = NULL;
		m_right = NULL;
		//delete[]m_component;
		//m_component = NULL;
	}
};

//ÓÃÓÚ¼ì²éÓÃÑù±¾;
//»ìÏý¾ØÕóÐÐÎªÊµ¼Ê¹éÊô£¬ÁÐÎªÊµÑé¹éÊô;
struct checkData
{
	double *m_rate;//¼ì²éÕýÈ·ÂÊ;
	Eigen::MatrixXd** m_checkData;//¶ÔÓ¦µÄÃ¿Ò»¸öÊý¾Ý¾ØÕó;
	int m_n;//·ÖÀàÊý;
	Eigen::MatrixXi* m_confusionMatrix;
	double kappa;
	checkData(int n) :m_n(n)
	{
		m_rate = new double[n];
		m_checkData = new Eigen::MatrixXd*[n];
		m_confusionMatrix = new Eigen::MatrixXi(m_n, m_n);
		for (int row = 0; row < m_n;row++)
		{
			for (int col = 0; col < m_n;col++)
			{
				(*m_confusionMatrix)(row, col) = 0;
			}
		}
	}
	~checkData()
	{
		delete[]m_rate;
		delete m_confusionMatrix;
		for (int i = 0; i < m_n;i++)
		{
			delete m_checkData[i];
		}
		delete[]m_checkData;
	}

	//ÀûÓÃÒÑ¾­·ÖÀàºÃµÄ»ìÏý¾ØÕó¼ÆËãKappaÖµ;
	//º¯Êý²ÎÊýnDataÎªÊý¾Ý×ÜÊý;
	void calculateKappa(int nData)
	{
		double P0 = m_confusionMatrix->trace() / (double( nData));
		double Pe = 0;
		for (int i = 0; i < m_n;i++)
		{
			double rowSum = (m_confusionMatrix->rowwise().sum())(i);
			double colSum = (m_confusionMatrix->colwise().sum())(i);
			Pe += (rowSum*colSum / nData);
		}
		Pe = Pe / nData;
		kappa = (P0 - Pe) / (1 - Pe);
		//std::cout << "num of datas: " << nData << std::endl;
		//std::cout << "kappa: " << kappa << std::endl;
	}
};

struct imageData
{
	int m_nBand;//²¨¶ÎÊý;
	int m_xSize;
	int m_ySize;
	double **m_imageData;
	imageData(int nBand, int xSize, int ySize) :m_nBand(nBand), m_xSize(xSize), m_ySize(ySize)
	{
		m_imageData = new double*[nBand];
		for (int i = 0; i < nBand;i++)
		{
			m_imageData[i] = new double[xSize*ySize];
		}
	}
	//ÊµÑéÓÃÊä³ö£¬bandÎª´Ó1ÆðËã;
	void output(int band)
	{
		if (m_nBand<band)
		{
			return;
		}
		for (int i = 0; i < m_xSize*m_ySize;i++)
		{
			std::cout << (m_imageData[band - 1])[i] << "  ";
		}
		std::cout << std::endl;
	}
	~imageData()
	{
		for (int i = 0; i < m_nBand; i++)
		{
			delete[](m_imageData[i]);
		}
		delete[] m_imageData;
	}
};

int main()
{
	std::string imageName = "quickbird2.jpg";
	std::string path = "E://RS_IMAGE//";
	std::string fileName = "quickbird2train.txt";
	std::string imagePath = path + imageName;
	std::string filePath = path + fileName;
	const char* imagePathChar = imagePath.c_str();
	GDALAllRegister();
	GDALDataset*pDataset;
	pDataset = (GDALDataset*)GDALOpen(imagePathChar, GA_ReadOnly);
	if (pDataset==NULL)
	{
		std::cout << "²»ÄÜ¶ÁÈ¡" << std::endl;
		return 0;
	}
	int nBand = pDataset->GetRasterCount();
	int xSize = pDataset->GetRasterXSize();
	int ySize = pDataset->GetRasterYSize();
	imageData* pData = new imageData(nBand, xSize, ySize);
	for (int i = 0; i < nBand;i++)
	{
		GDALRasterBand* pBand = pDataset->GetRasterBand(i + 1);
		pBand->RasterIO(GF_Read, 0, 0, xSize, ySize, (pData->m_imageData)[i], xSize, ySize,
			GDT_Float64,0,0);

	}

	int nClassify;//Ò»¹²µÄÖÖÀà;
	classify* data = loadFile(filePath, nClassify);
	std::cout << "finish loading file" << std::endl;
	Eigen::MatrixXd** trainClassifies = new Eigen::MatrixXd*[nClassify];
	checkData* pCheckData = new checkData(nClassify);
	//allDatasÎª×ÜÑù±¾£¨ºóÆÚ½øÐÐ¼ìºË£©,trainClassifiesÎª½øÐÐÑµÁ·µÄÑù±¾;
	for (int i = 0; i < nClassify;i++)
	{
		readData(data + i, pData,&(trainClassifies[i]),&((pCheckData->m_checkData)[i]));
	}
	GMM** trains = new GMM*[nClassify];
	for (int i = 0; i < nClassify;i++)
	{
		trains[i] = new GMM(data[i].m_component, nBand);
		(trains[i])->setData(trainClassifies[i]);
		trains[i]->practiceModel();
		std::cout << "finish practicing " << i << " model" << std::endl;
		//trains[i]->output();
	}
	std::cout << "finish practicing all models" << std::endl;
	int nData = (pData->m_xSize)*(pData->m_ySize);
	
	//¼ÇÂ¼Ã¿¸öÏñËØµãËùÊôÀà±ð;
	Eigen::MatrixXi* resultMatrix = classifyData(trains, pData->m_imageData, nData,
		pData->m_nBand, nClassify, pData->m_xSize, pData->m_ySize);
	std::cout << "finish classifying data" << std::endl;
	checkDataF(trains, nClassify, nBand, pCheckData);
	std::cout << "finish checking data" << std::endl;
	std::string outputPath = path + "outputImage.tif";
	outputImage2(outputPath, resultMatrix,nClassify);
	std::cout << "finish outputing image" << std::endl;
	outputPath = path + "outputCheck.txt";
	outputCheck(pCheckData, outputPath);
	for (int i = 0; i < nClassify;i++)
	{
		data[i].deleteData();
		delete trainClassifies[i];
		delete trains[i];
	}
	delete pData;
	pData = NULL;
	GDALClose((GDALDatasetH)pDataset);
	delete[]trains;
	delete[]trainClassifies;
	delete pCheckData;
	delete[]data;
	delete resultMatrix;
	resultMatrix = NULL;
	data = NULL;
}

//´Ópath´¦¼ÓÔØÎÄ¼þ;
//nÎªÒ»¹²¼ÓÔØµÄÖÖÀàÊý;
//·µ»ØµÄÖ¸ÕëÎª¹¹ÔìµÄclassifyÊý×é;
classify* loadFile(std::string path,int &n)
{
	std::ifstream myfile(path);
	if (!myfile)
	{
		std::cout << "read file error" << std::endl;
		return NULL;
	}
	std::string line;
	myfile >> n;
	classify* classifies = new classify[n];
	for (int i = 0; i < n;i++)
	{
		myfile >> classifies[i].m_n >> classifies[i].m_component;
		int n_areas = classifies[i].m_n;
		classifies[i].m_down = new int[n_areas];
		classifies[i].m_left = new int[n_areas];
		classifies[i].m_right = new int[n_areas];
		classifies[i].m_up = new int[n_areas];
		for (int j = 0; j < classifies[i].m_n;j++)
		{
			myfile >> classifies[i].m_up[j] >> classifies[i].m_down[j]
				>> classifies[i].m_left[j] >> classifies[i].m_right[j];
			//std::cout << classifies[i].m_up[j] << " " << classifies[i].m_down[j]
				//<< " " << classifies[i].m_left[j] << " " << classifies[i].m_right[j] << std::endl;
		}
	}
	myfile.close();
	return classifies;
}

//´«ÈëÐèÒª¶ÁÈ¡µÄclassifyµÄÖ¸Õë,¹¹ÔìÊý¾Ý¾ØÕó,
//ÆäÖÐËæ»úÑ¡ÔñÊý¾ÝÁ¿µÄÎå·ÖÖ®Ò»ÎªÑµÁ·Ñù±¾;
//trainMatrixÖ¸ÏòÑµÁ·Ñù±¾,allMatrix¶ÔÓ¦¼ìºËÑù±¾;
void readData(classify* classifies,imageData* pData,Eigen::MatrixXd** trainMatrix,
	Eigen::MatrixXd** allMatrix)
{
	if (!classifies)
	{
		return;
	}
	int nArea = classifies->m_n;
	int xSize = pData->m_xSize;
	int ySize = pData->m_ySize;
	int nData = 0;//Ò»¹²µÄÑµÁ·µãÊýÄ¿;
	int nBand = pData->m_nBand;
	for (int i = 0; i < nArea;i++)
	{
		nData += (((classifies->m_down)[i] - (classifies->m_up)[i] + 1)*((classifies->m_right)[i] - (classifies->m_left)[i] + 1));
	}
	Eigen::MatrixXd* pDataMatrix = new Eigen::MatrixXd(nData, pData->m_nBand);
	int n_trainData = nData / TRAINRATE;
	Eigen::MatrixXd* pTrainMatrix = new Eigen::MatrixXd(n_trainData, pData->m_nBand);
	for (int band = 0; band < nBand;band++)
	{
		double *firstData = (pData->m_imageData)[band];
		int imageIndex = 0;//µ±Ç°ÕýÔÚ¶ÔimageIndexÏñËØ½øÐÐ²Ù×÷;
		for (int i = 0; i < nArea; i++)
		{
			int firstRow = (classifies->m_up)[i];
			int lastRow = (classifies->m_down)[i];
			int firstCol = (classifies->m_left)[i];
			int lastCol = (classifies->m_right)[i];
			//firstData = firstData + (xSize*firstRow+fir)
			for (int row = firstRow; row <= lastRow;row++)
			{
				//firstData = firstData + (xSize*row) + firstCol;
				double *tmpData = firstData + (xSize*row) + firstCol;
				for (int col = firstCol; col <= lastCol;col++)
				{
					(*pDataMatrix)(imageIndex, band) = (*(tmpData + col - firstCol));
					imageIndex++;
				}
			}
		}
	}
	for (int i = 0; i < n_trainData;i++)
	{
		int trainIndex = rand() % (nData - 1);
		pTrainMatrix->row(i) = pDataMatrix->row(trainIndex);
	}
	*trainMatrix = pTrainMatrix;
	*allMatrix = pDataMatrix;
}

//ÀûÓÃÒÑ¾­±»ÑµÁ·Ñù±¾ÑµÁ·ºÃµÄ·ÖÀàÆ÷¶ÔÍ¼ÏñÊý¾Ý½øÐÐ·ÖÀà;
//nDataÎªÏñËØµã¸öÊý£¬nBandÎªÊý¾ÝÎ¬¶È£¬nClassifyÎª·ÖÀàÀà±ðÊý;
//·µ»ØÖµÃ¿¸öÏñËØÊôÓÚµÄÀà±ð;
Eigen::MatrixXi* classifyData(GMM** models,double **image,int nData,int nBand,int nClassify,
	int xSize,int ySize)
{
	Eigen::MatrixXd *dataMatrix = new Eigen::MatrixXd(nData, nBand);
	Eigen::MatrixXi *resultMatrix = new Eigen::MatrixXi(ySize, xSize);
	for (int i = 0; i < nBand;i++)
	{
		double *pData = image[i];
		for (int j = 0; j < nData;j++)
		{
			(*dataMatrix)(j, i) = *(pData);
			pData++;
		}
	}
	Eigen::MatrixXd* probMatrix = new Eigen::MatrixXd(nData, nClassify);
	for (int i = 0; i < nClassify;i++)
	{
		GMM* pModel = models[i];
		pModel->calculateProbility(dataMatrix, probMatrix, i);
	}
	std::cout << "finish calculating probability matrix" << std::endl;
	for (int row = 0; row < ySize;row++)
	{
		for (int col = 0; col < xSize;col++)
		{
			int index = row*xSize + col;
			int maxIndex;
			probMatrix->row(index).maxCoeff(&maxIndex);
			(*resultMatrix)(row, col) = maxIndex;
		}
	}
	delete dataMatrix;
	delete probMatrix;
	return resultMatrix;
}

//ÊÊÓÃÓÚÓÐÎåÖÖ·ÖÀàÈý¸ö²¨¶ÎµÄÇé¿ö;
void outputImage(std::string fileName, Eigen::MatrixXi* resultMatrix)
{
	int xSize = resultMatrix->cols();
	int ySize = resultMatrix->rows();
	GDALDriver *pDriver;
	GDALDataset *pDataset;
	pDriver = GetGDALDriverManager()->GetDriverByName("Gtiff");
	const char* outputFile = fileName.c_str();
	pDataset = pDriver->Create(outputFile, xSize, ySize, 3, GDT_Byte, NULL);
	unsigned char** newImageData = new unsigned char*[3];
	for (int i = 0; i < 3;i++)
	{
		newImageData[i] = new unsigned char[xSize*ySize];
	}
	for (int row = 0; row < ySize;row++)
	{
		for (int col = 0; col < xSize;col++)
		{
			int result = (*resultMatrix)(row, col);
			switch (result)
			{
			case(0) :
			{
				newImageData[0][row*xSize + col] = 0;
				newImageData[1][row*xSize + col] = 0;
				newImageData[2][row*xSize + col] = 0;
				break;
			}
			case(1) :
			{
				newImageData[0][row*xSize + col] = 255;
				newImageData[1][row*xSize + col] = 255;
				newImageData[2][row*xSize + col] = 255;
				break;
			}
			case(2) :
			{
				newImageData[0][row*xSize + col] = 255;
				newImageData[1][row*xSize + col] = 0;
				newImageData[2][row*xSize + col] = 0;
				break;
			}
			case(3) :
			{
				newImageData[0][row*xSize + col] = 0;
				newImageData[1][row*xSize + col] = 255;
				newImageData[2][row*xSize + col] = 0;
				break;
			}
			case(4) :
			{
				newImageData[0][row*xSize + col] = 0;
				newImageData[1][row*xSize + col] = 0;
				newImageData[2][row*xSize + col] = 255;
				break;
			}
			default:
				break;
			}
		}
	}
	for (int i = 0; i < 3;i++)
	{
		GDALRasterBand *pBand = pDataset->GetRasterBand(i + 1);
		pBand->RasterIO(GF_Write, 0, 0, xSize, ySize, newImageData[i], xSize, ySize,
			GDT_Byte, 0, 0);
	}
	GDALClose((GDALDatasetH)pDataset);
	for (int i = 0; i < 3;i++)
	{
		delete [](newImageData[i]);
	}
	delete[]newImageData;
	pDataset = NULL;
}

//ÊÊÓÃÓÚ¶àÓÚÎåÖÖ·ÖÀàµÄÇé¿ö£¬ÒÀ¾ÉÊÇÈý¸ö²¨¶Î;
void outputImage2(std::string fileName, Eigen::MatrixXi* resultMatrix, int nClassify)
{
	if (nClassify<=5)
	{
		outputImage(fileName, resultMatrix);
		return;
	}
	Eigen::MatrixXi* colorTable = new Eigen::MatrixXi(3, nClassify);
	//colorTable,´ÓÀïÃæÑ¡Ôñ¶ÔÓ¦ÖÖÀàµÄÑÕÉ«;
	(*colorTable)(0, 0) = 0;
	(*colorTable)(1, 0) = 0;
	(*colorTable)(2, 0) = 0;
	(*colorTable)(0, 1) = 255;
	(*colorTable)(1, 1) = 255;
	(*colorTable)(2, 1) = 255;
	(*colorTable)(0, 2) = 255;
	(*colorTable)(1, 2) = 0;
	(*colorTable)(2, 2) = 0;
	(*colorTable)(0, 3) = 0;
	(*colorTable)(1, 3) = 255;
	(*colorTable)(2, 3) = 0;
	(*colorTable)(0, 4) = 0;
	(*colorTable)(1, 4) = 0;
	(*colorTable)(2, 4) = 255;
	int nLeft = nClassify - 5;
	for (int i = 0; i < nLeft;i++)
	{
		(*colorTable)(0, 4 + i) = rand() % 255;
		(*colorTable)(1, 4 + i) = rand() % 255;
		(*colorTable)(2, 4 + i) = rand() % 255;
	}
	//colorTableÉèÖÃÍê±Ï;
	int xSize = resultMatrix->cols();
	int ySize = resultMatrix->rows();
	GDALDriver *pDriver;
	GDALDataset *pDataset;
	pDriver = GetGDALDriverManager()->GetDriverByName("Gtiff");
	const char* outputFile = fileName.c_str();
	pDataset = pDriver->Create(outputFile, xSize, ySize, 3, GDT_Byte, NULL);
	unsigned char** newImageData = new unsigned char*[3];
	for (int i = 0; i < 3; i++)
	{
		newImageData[i] = new unsigned char[xSize*ySize];
	}
	for (int row = 0; row < ySize; row++)
	{
		for (int col = 0; col < xSize; col++)
		{
			int result = (*resultMatrix)(row, col);
			newImageData[0][row*xSize + col] = (*colorTable)(0, result);
			newImageData[1][row*xSize + col] = (*colorTable)(1, result);
			newImageData[2][row*xSize + col] = (*colorTable)(2, result);
		}
	}
	for (int i = 0; i < 3; i++)
	{
		GDALRasterBand *pBand = pDataset->GetRasterBand(i + 1);
		pBand->RasterIO(GF_Write, 0, 0, xSize, ySize, newImageData[i], xSize, ySize,
			GDT_Byte, 0, 0);
	}
	GDALClose((GDALDatasetH)pDataset);
	for (int i = 0; i < 3; i++)
	{
		delete[](newImageData[i]);
	}
	delete[]newImageData;
	pDataset = NULL;
	delete colorTable;
	colorTable = NULL;
}

//ÀûÓÃÒÑ¾­ÑµÁ·ºÃµÄ·ÖÀàÆ÷¶ÔcheckData½øÐÐ·ÖÀà£¬²¢½øÐÐ»ìÏý¾ØÕóºÍkappaÖµµÄ¼ÆËã;
void checkDataF(GMM** models,int nClassify,int nBand,checkData* pCheckData)
{
	int sumData = 0;
	Eigen::MatrixXi* pConfusionMatrix = pCheckData->m_confusionMatrix;
	for (int i = 0; i < nClassify;i++)
	{
		int nCheckData = (pCheckData->m_checkData)[i]->rows();
		sumData += nCheckData;
		Eigen::MatrixXd *pProbMatrix = new
			Eigen::MatrixXd(nCheckData, nClassify);
		for (int j = 0; j < nClassify;j++)
		{
			GMM* pModel = models[j];
			pModel->calculateProbility((pCheckData->m_checkData)[i], pProbMatrix, j);
		}
		for (int index = 0; index < nCheckData;index++)
		{
			int maxCol;
			pProbMatrix->row(index).maxCoeff(&maxCol);
			(*pConfusionMatrix)(i, maxCol) = ((*pConfusionMatrix)(i, maxCol)) + 1;
		}
		delete pProbMatrix;
	}
	//std::cout << *(pConfusionMatrix) << std::endl;
	pCheckData->calculateKappa(sumData);
}

//¶ÔcheckData½øÐÐÊä³ö;
//Êä³ö»ìÏý¾ØÕóºÍkappaÏµÊý;
void outputCheck(checkData* pCheckData,std::string pathFile)
{
	std::cout << "begin outputting check report" << std::endl;
	std::ofstream outPut(pathFile);
	if (!outPut)
	{
		std::cout << "output check error" << std::endl;
	}
	outPut << *(pCheckData->m_confusionMatrix) << std::endl;
	outPut << "kappa: " << pCheckData->kappa << std::endl;
	outPut.close();
}
