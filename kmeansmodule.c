#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>




double EDistance(double *point1, double *point2,int d);
int converge(double **prev ,double **curr, int curr_itr,int Max_itr, int K , int d, double epsilon);
double** assign(double** data,double **currClusters,int d,int K, int N);
void freeArray(double** centroids, int len);
PyObject* fit(double** centroids, double** data,int d,int k,double eps,int Max_iter, int DataSize);
void handleMemoryFail();


double** convert_array(PyObject *pyData,int d);
double* get_point(PyObject *pyPoint,int d);


void handleMemoryFail() {
    printf("Failed allocating memory");
    exit(1);
}

double EDistance(double *point1, double *point2,int d){
    int i;
    double sum = 0.0;
    double dist;
    for (i = 0; i < d; i++){
        dist = (point1[i] - point2[i]);
        sum += (dist * dist);
    }
    return sum;
}



int converge(double **prev ,double **curr, int curr_itr,int Max_itr, int K , int d, double epsilon){
    int i;
    if (curr_itr >= Max_itr) return 1;
       for (i = 0 ;i < K; i++){
           if (EDistance(prev[i],curr[i],d) > (epsilon * epsilon)) return 0;
    }
    return 1;
    }


double** assign(double** data, double **currClusters,int d,int K, int N){
    int i,j;
    int index;
    double minDist , dist;
    double* data_entry;
    int* assignments_cnt = (int*)calloc(K ,sizeof(int));
    double** assignments = (double**)calloc(K ,sizeof(double*));

    if (assignments_cnt == NULL || assignments == NULL) handleMemoryFail();

    for (i = 0; i < K; i++) {
        assignments[i] = (double*)calloc(d,sizeof(double));
        if (assignments[i] == NULL) handleMemoryFail();
    }

    //Assign every x_i to the closest current cluster
    for (i = 0; i < N ; i++){
        data_entry = data[i];
        minDist = DBL_MAX;
        index = -1;
        for (j = 0 ; j < K ; j++){
            dist = EDistance(data_entry,currClusters[j],d);
            if (dist < minDist) {
                minDist = dist;
                index = j;
            }
        }
        assignments_cnt[index] += 1;
        for (j = 0; j < d; j++) {
            assignments[index][j] += data_entry[j];
        }
        }

        for (i = 0; i < K; i++) {
            for (j = 0; j < d; j++) {
                assignments[i][j] = assignments[i][j] / assignments_cnt[i];
        }
    }
    free(assignments_cnt);
    return assignments;
}


void freeArray(double** arr, int len) {
    int i;
    for (i = 0; i < len; i++) {
        free(arr[i]);
    }
    free(arr);
}



PyObject* fit(double** centroids, double** data, int d, int k, double eps,int Max_iter, int DataSize){
    int i,j;
    PyObject* final_centroids;
    PyObject* point;
    PyObject* python_float;

    int is_converged = 0; int cnt = 0 ;
    double** prevCentroids; double** newCentroids = centroids;


    while(is_converged == 0){
        prevCentroids = newCentroids;
        newCentroids = assign(data,prevCentroids,d,k,DataSize);

        cnt++;
        is_converged = converge(prevCentroids,newCentroids,cnt,Max_iter,k,d,eps);

        freeArray(prevCentroids,k);
    }

    final_centroids = PyList_New(k);

    for (i = 0; i < k ; i++){
        point = PyList_New(d);
        for (j = 0; j < d; j++){
            python_float = Py_BuildValue("d", newCentroids[i][j]);
            PyList_SetItem(point, j, python_float);
        }
        PyList_SetItem(final_centroids, i, point);
    }

    freeArray(data,DataSize);
    freeArray(newCentroids,k);
    return final_centroids;

}


static PyObject* fit_c(PyObject *self, PyObject *args)
{
    PyObject *pyCentroids;
    PyObject *pyData;
    int d, k,iter;
    double eps;
    double** data;
    double** centroids;

    if(!PyArg_ParseTuple(args, "OOiiid", &pyCentroids, &pyData, &d, &k,&iter ,&eps)) {
        return NULL; /* In the CPython API, a NULL value is never valid for a
                        PyObject* so it is used to signal that an error has occurred. */
    }
    data = convert_array(pyData,d);
    centroids = convert_array(pyCentroids,d);
    int n = (int)PyList_Size(pyData);

/* This builds the answer ("O" = Convert a C Object to a Python raw object) */
    return fit(centroids,data,d,k,eps,iter,n); /*  Py_BuildValue(...) returns a PyObject*  */
}


static PyMethodDef fitMethods[] = {
    {"fit_c",                   /* the Python method name that will be used */
      (PyCFunction) fit_c, /* the C-function that implements the Python function and returns static PyObject*  */
      METH_VARARGS,           /* flags indicating parameters
accepted for this function */
      PyDoc_STR("fit expects : (double** centroids,double** data ,int d ,int k ,double eps, int N)")}, /*  The docstring for the function */
    {NULL, NULL, 0, NULL}     /* The last entry must be all NULL as shown to act as a
                                 sentinel. Python looks for this entry to know that all
                                 of the functions for the module have been defined. */
};

static struct PyModuleDef kmeansmodule = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,  /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    fitMethods /* the PyMethodDef array from before containing the methods of the extension */
};


PyMODINIT_FUNC PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&kmeansmodule);
    if (!m) {
        return NULL;
    }
    return m;
}


double** convert_array(PyObject *pyData,int d){
    double** res;
    Py_ssize_t i,n;
    PyObject* point;
    double* c_point;
    n = PyList_Size(pyData);
    res = (double**)malloc(n * sizeof(double*));
    if (!res) handleMemoryFail();

    for(i = 0 ; i < n; i++){
        point = PyList_GetItem(pyData, i);
        c_point = get_point(point,d);
        res[i] = c_point;
    }
    return res;
}


double* get_point(PyObject *pyPoint,int d){
    int j;
    PyObject* entry;
    double* c_point;
    c_point = (double*)calloc(d, sizeof(double));
    if (c_point == NULL) handleMemoryFail();
    for(j = 0 ; j < d ; j++){
        entry = PyList_GetItem(pyPoint, j);
        c_point[j] = PyFloat_AsDouble(entry);
    }
    return c_point;
}
