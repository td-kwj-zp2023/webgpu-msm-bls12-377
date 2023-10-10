import * as cuZK from "../submission/cuzk/cuzk"
import React, { useEffect } from 'react';
import mustache from 'mustache'

export const CUZK: React.FC = () => {
    useEffect(() => {
        async function execute_cuZK() {
            console.log("Implement cuZK!")        
        }
        execute_cuZK();
    }, []);

    return (
        <div>
        </div>
    );
}
